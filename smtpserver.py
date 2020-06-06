import smtplib
import json

with open("credentials.json") as data:
    credientials = json.load(data)

# Senders email
sender_email = credientials["sender_email"]
# Receivers email
rec_email = credientials["reciever_email"]
# Senders password
password = credientials['sender_password']


def email_msg(message):
    # Initialize the server variable
    server = smtplib.SMTP('smtp.gmail.com', 587)
    # Start the server connection
    server.starttls()
    # Login
    server.login(sender_email, password)
    print("Login Success!")
    # Send Email
    server.sendmail(sender_email, rec_email, message)
    server.quit()
