import sqlite3

# conn = sqlite3.connect("Chatbot/chatbot.db")
# cursor = conn.cursor()

# cursor.execute("DELETE from che")
# cursor.execute("DELETE from writes")

# conn.commit()
# conn.close()

# print("Test threads removed successfully!")


conn = sqlite3.connect("Chatbot/chatbot.db")
cursor = conn.cursor()

cursor.execute("DELETE from long_term_memory")

conn.commit()
conn.close()

print("Test threads removed successfully!")