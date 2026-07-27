import sqlite3

conn = sqlite3.connect("Chatbot/chatbot.db")
cursor = conn.cursor()

cursor.execute("DELETE from checkpoints")
cursor.execute("DELETE from writes")

conn.commit()
conn.close()

print("Test threads removed successfully!")