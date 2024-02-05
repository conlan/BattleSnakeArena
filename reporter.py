import requests
import json

class Reporter():
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def report(self, mean_validation_reward):
        print(f'\nMEAN VALIDATION REWARD = : {mean_validation_reward}')

        self._report_to_discord({
            "mean_validation_reward" : mean_validation_reward
        })

    def _report_to_discord(self, data):
        mean_validation_reward = data["mean_validation_reward"]

        discord_message = "**Validation Report:**\n\n"
        discord_message += f":moneybag:   **Mean Validation Reward**: {mean_validation_reward}\n\n"
        discord_message += "------------------------------------------------------------\n\n"

        payload = {
            "content": discord_message
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(self.webhook_url, data=json.dumps(payload), headers=headers)

        if response.status_code == 204:
            print("Message sent successfully to discord")
        else:
            print(f"Failed to send message. Status code: {response.status_code}")