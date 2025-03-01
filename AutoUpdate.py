import os
import time
import subprocess
import git  # type: ignore
import requests
import sys
from datetime import datetime

# Configuration - modify these variables according to your setup
GITHUB_REPO_URL = "https://github.com/JMUNX/QUARGLE-BOT"  # Your GitHub repository URL
LOCAL_PATH = "\QUARGLE-BOT"  # Where you want to store the bot files locally
SCRIPT_NAME = "QUARGLE-V5.py"  # Name of your main bot script
CHECK_INTERVAL = 10  # Check every 5 minutes (in seconds)
BRANCH = "main"  # Branch to monitor


class BotUpdater:
    def __init__(self):
        self.process = None
        self.last_commit = None

        # Ensure local directory exists
        if not os.path.exists(LOCAL_PATH):
            os.makedirs(LOCAL_PATH)
            self.clone_repo()
        else:
            self.repo = git.Repo(LOCAL_PATH)

    def clone_repo(self):
        """Clone the repository if it doesn't exist"""
        print(f"Cloning repository from {GITHUB_REPO_URL}")
        self.repo = git.Repo.clone_from(GITHUB_REPO_URL, LOCAL_PATH)
        self.last_commit = self.get_current_commit()

    def get_current_commit(self):
        """Get the current commit hash"""
        return self.repo.head.commit.hexsha

    def get_latest_commit_from_github(self):
        """Get the latest commit hash from GitHub API"""
        repo_name = GITHUB_REPO_URL.split("github.com/")[1].replace(".git", "")
        api_url = f"https://api.github.com/repos/{repo_name}/commits/{BRANCH}"

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            return response.json()["sha"]
        except requests.RequestException as e:
            print(f"Error checking GitHub: {e}")
            return None

    def update_bot(self):
        """Pull latest changes from GitHub"""
        print(f"[{datetime.now()}] Updating bot...")
        try:
            self.repo.remotes.origin.pull()
            return True
        except git.GitCommandError as e:
            print(f"Error pulling updates: {e}")
            return False

    def stop_bot(self):
        """Stop the running bot process"""
        if self.process and self.process.poll() is None:
            print(f"[{datetime.now()}] Stopping bot...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def start_bot(self):
        """Start the bot process"""
        print(f"[{datetime.now()}] Starting bot...")
        bot_path = os.path.join(LOCAL_PATH, SCRIPT_NAME)
        self.process = subprocess.Popen(
            [sys.executable, bot_path],
            cwd=LOCAL_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def check_for_updates(self):
        """Main update checking loop"""
        # Start bot initially
        self.start_bot()

        while True:
            try:
                latest_commit = self.get_latest_commit_from_github()

                if latest_commit and latest_commit != self.last_commit:
                    print(f"[{datetime.now()}] New update detected!")
                    self.stop_bot()
                    if self.update_bot():
                        self.last_commit = latest_commit
                        self.start_bot()
                    else:
                        print(
                            f"[{datetime.now()}] Update failed, restarting current version"
                        )
                        self.start_bot()

                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                print(f"[{datetime.now()}] Error in update loop: {e}")
                time.sleep(CHECK_INTERVAL)


def main():
    print(f"[{datetime.now()}] Starting bot updater...")
    updater = BotUpdater()
    updater.check_for_updates()


if __name__ == "__main__":
    main()
