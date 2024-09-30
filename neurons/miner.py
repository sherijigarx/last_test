# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 <ETG>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Bittensor Miner lib:

# Step 1: Import necessary libraries and modules
from scipy.io.wavfile import write as write_wav
import bittensor as bt
import numpy as np
import torchaudio
import traceback
import argparse
import typing
import torch
import wave
import time
import sys
import os


# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

# import this repo
from models.text_to_music import MusicGenerator
import lib.protocol
import lib


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--music_model", default='facebook/musicgen-medium' , help="The model to be used for Music Generation." 
    )
    parser.add_argument(
        "--music_path", default=None , help="The Finetuned model to be used for Music Generation." 
    )

    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=50, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


def main(config):
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Check the supplied model and log the appropriate information.
    try:
        # Assuming `config` is an object holding command-line arguments
        if config.music_path:
            bt.logging.info(f"Using the custom model path for Text-To-Music: {config.music_path}")
            ttm_models = MusicGenerator(model_path=config.music_path)
        elif config.music_model in ["facebook/musicgen-medium", "facebook/musicgen-large"]:
            bt.logging.info(f"Using the Text-To-Music with the supplied model: {config.music_model}")
            ttm_models = MusicGenerator(model_path=config.music_model)
        else:
            bt.logging.error(f"Wrong model supplied for Text-To-Music: {config.music_model}")
            exit(1)
    except Exception as e:
        bt.logging.info(f"An error occurred while model initilization: {e}")
        exit(1)

    bt.logging.info("Setting up bittensor objects.")
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    # The blacklist function decides if a request should be ignored.
    def music_blacklist_fn(synapse: lib.protocol.MusicGeneration) -> typing.Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        elif synapse.dendrite.hotkey in metagraph.hotkeys and metagraph.S[metagraph.hotkeys.index(synapse.dendrite.hotkey)] < lib.MIN_STAKE:
            # Ignore requests from entities with low stake.
            bt.logging.trace(
                f"Blacklisting hotkey {synapse.dendrite.hotkey} with low stake"
            )
            return True, "Low stake"
        elif synapse.dendrite.hotkey in lib.BLACKLISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting Key recognized as blacklisted hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Blacklisted hotkey"
        elif synapse.dendrite.hotkey in lib.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return False, "Hotkey recognized!"
        else:
            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Hotkey recognized as Blacklisted!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def music_priority_fn(synapse: lib.protocol.MusicGeneration) -> float:
        caller_uid = metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def convert_music_to_tensor(audio_file):
        '''Convert the audio file to a tensor'''
        try:
            # Get the file extension
            _, file_extension = os.path.splitext(audio_file)

            if file_extension.lower() in ['.wav', '.mp3']:
                # load the audio file
                audio, sample_rate = torchaudio.load(audio_file)
                # convert the audio file to a tensor/list
                audio = audio[0].tolist()
                return audio
            else:
                bt.logging.error(f"Unsupported file format: {file_extension}")
                return None
        except Exception as e:
            bt.logging.error(f"An error occurred while converting the file: {e}")

    def ProcessMusic(synapse: lib.protocol.MusicGeneration) -> lib.protocol.MusicGeneration:
        bt.logging.info(f"Generating music with the model: {config.music_model}")
        music = ttm_models.generate_music(synapse.text_input, synapse.duration)

        # Check if 'music' contains valid audio data
        if music is None:
            bt.logging.error("No music generated!")
            return None
        else:
            try:
                sampling_rate = 32000
                # Assuming write_wav function exists and works as intended
                write_wav("musicgen_out.wav", rate=sampling_rate, data=music) # synapse.dendrite.hotkey + "_musicgen_out.wav"
                bt.logging.success(f"Text to Music has been generated! and saved to: musicgen_out.wav")
                # Assuming convert_music_to_tensor function exists to convert WAV to tensor
                music_tensor = convert_music_to_tensor("musicgen_out.wav")
                synapse.music_output = music_tensor
                return synapse
            except Exception as e:
                bt.logging.error(f"An error occurred while processing music output: {e}")
                return None


####################################################### Attach Axon  ##############################################################
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn= ProcessMusic,
        blacklist_fn= music_blacklist_fn,
        priority_fn= music_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon {ProcessMusic} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # Below: Periodically update our knowledge of the network graph.
            if step % 500 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[my_subnet_uid]:.6f} | "
                    f"Rank:{metagraph.R[my_subnet_uid]:.6f} | "
                    f"Trust:{metagraph.T[my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[my_subnet_uid]:.6f} | "
                    f"Incentive:{metagraph.I[my_subnet_uid]:.6f} | "
                    f"Emission:{metagraph.E[my_subnet_uid]}"
                )
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
# Entry point for the script
if __name__ == "__main__":
    config = get_config()
    main(config)
