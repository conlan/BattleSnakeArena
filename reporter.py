import os
import requests
import json

class Reporter():
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

        self.history = []
        self.save_path = None

    def load_history(self, path) -> None:
        self.save_path = path

        # if file exists, load it
        if (os.path.exists(path)):
            with open(path, 'r') as file:
                self.history = json.load(file)

            print(f"History loaded from {path} containing {len(self.history)} records")
        else:
            print(f"History file not found at {path}. Starting new history")

    def save_history(self) -> None:
        # save history to json path
        json.dump(self.history, open(self.save_path, 'w'))

        print(f"\n    SAVED history to {self.save_path}")

    def report(self, validation_results, opponent, curr_step):
        mean_validation_reward = validation_results["mean_validation_reward"]

        print(f'\nMEAN VALIDATION REWARD = {mean_validation_reward}')        
        
        history_obj = {
            "opponent" : opponent,
            "validation_results" : validation_results,
            "curr_step" : curr_step
        }

        self._report_to_discord(history_obj)

        self.history.append(history_obj)

    def _report_to_discord(self, data):
        validation_results = data["validation_results"]
        
        mean_validation_reward = validation_results["mean_validation_reward"]
        win_rate = validation_results["win_rate"]

        discord_message = "**Validation Report:**\n\n"
        
        discord_message += f":snake:   **Opponent**: {data['opponent']}\n\n"
        discord_message += f":chart_with_upwards_trend:   **Win Rate**: {win_rate}%\n\n"
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