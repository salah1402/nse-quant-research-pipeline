import time
import random
import requests

# --- CONFIGURATION ---
BOT_TOKEN = "8782797872:AAEchAgMyqm4aSMdmujyjFtfOy5rPhfZ_g0" # Replace if you revoked it
CHAT_ID = "5336903768"

def send_telegram_alert(message):
    """Sends a message to your Telegram bot."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Alert sent to Telegram!")
        else:
            print(f"❌ Failed to send alert. Error: {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")

# --- MOCK MARKET DATA ---
print("🚀 Mock Scanner Started. Simulating RELIANCE...")
print("Waiting for ORB (9:15-9:30)...")
time.sleep(2) # Simulate ORB building

orb_high = 2500.00
orb_low = 2480.00
print(f"ORB Marked: High={orb_high}, Low={orb_low}")

# Simulate price movement
current_price = 2495.00
print(f"Current Price: {current_price}")

# 1. Simulate Breakout
time.sleep(2)
current_price = 2505.00
print(f"🚨 Breakout! Price moved to {current_price}")

# 2. Simulate Retest
time.sleep(2)
current_price = 2501.00
print(f"🔄 Retesting ORB High at {current_price}...")

# 3. Simulate Confirmation and Trigger Alert
time.sleep(2)
current_price = 2506.50
print(f"✅ Retest held! Confirmation candle closed at {current_price}")

alert_message = f"""
🚨 *ORB TRADE ALERT* 🚨

*Symbol:* RELIANCE
*Direction:* LONG
*Entry:* 2506.50
*Stop Loss:* 2498.00
*Target:* 2525.00
*Net R:R:* 2.1

_Validate on chart and place order manually._
"""

print("Sending alert...")
send_telegram_alert(alert_message)