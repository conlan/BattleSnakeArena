import os
import requests
import json

class Reporter():
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

        self.history = []
        self.save_path = None

    def load_history(self, path) -> dict:
        self.save_path = path

        # if file exists, load it
        if (os.path.exists(path)):
            with open(path, 'r') as file:
                self.history = json.load(file)

        print(f"History loaded from {path} containing {len(self.history)} records")

    def save_history(self) -> None:
        # save history to json path
        json.dump(self.history, open(self.save_path, 'w'))

        print(f"\nHistory saved to {self.save_path}\n")

    def report(self, mean_validation_reward, curr_step):
        print(f'\nMEAN VALIDATION REWARD = : {mean_validation_reward}')        

        history_obj = {
            "mean_validation_reward" : mean_validation_reward,
            "curr_step" : curr_step
        }

        self._report_to_discord(history_obj)

        self.history.append(history_obj)

    def _report_to_discord(self, data):
        mean_validation_reward = data["mean_validation_reward"]

        discord_message = "**Validation Report:**\n\n"
        
        discord_message += f":moneybag:   **Mean Reward**: {mean_validation_reward}\n\n"
        discord_message += f":stopwatch:   **Curr Step**: {data['curr_step']}\n\n"
        
        discord_message += "------------------------------------------------------------\n\n"

        if (self.webhook_url == None) or (self.webhook_url == ""):
            print(discord_message)
        else:
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