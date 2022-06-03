# I used miniforge for this installation
# https://github.com/conda-forge/miniforge
# Installation:
# conda install -c conda-forge netifaces
# pip install slack_sdk
import netifaces as ni
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from time import sleep
import syslog


#### All Constants
slack_bot_token = "xoxb-PUT_YOUR_TOKEN_HERE"
channel_address_slack = "#general"
wifi_interface = "wlan0"
logger_name = "Jetson Nano IP Sender:"
## End Constant


def send_slack_message():
    sent_msg = False
    while not sent_msg:
        try:
            client = WebClient(token=slack_bot_token)
            ip = ni.ifaddresses(wifi_interface)[ni.AF_INET][0]["addr"]
            message_text = f"Jetson Nano IP: {ip}"
            response = client.chat_postMessage(channel=channel_address_slack, text=message_text)
            assert response["message"]["text"] == message_text
            sent_msg = True
        except SlackApiError as e:
            syslog.syslog(syslog.LOG_CRIT, f"{logger_name} Got an error: {e.response['error']}")
            sleep(5)

        except Exception as e:
            syslog.syslog(syslog.LOG_CRIT, f"{logger_name} Got an error: {e}")
            sleep(5)


if __name__ == "__main__":
    send_slack_message()
