module.exports = {
    apps: [
      {
        name: "validator",
        script: "neurons/validator.py",
        args: "--netuid 31 --wallet.name valid --wallet.hotkey hot --subtensor.network test",
        interpreter: "python3", // Ensure this points to the correct Python interpreter if needed
      },
    ],
  };
  