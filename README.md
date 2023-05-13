# IntercomFace - face recognition system for intercom

## Goal
The IntercomFace system is an advanced facial recognition-based intercom system designed to provide secure and convenient access control for residential buildings. The system will use cutting-edge facial recognition technology to automatically identify and grant access to approved individuals. The primary goal of this project is to develop a reliable and user-friendly system compatible with any intercom systems with open video streaming, allowing for seamless integration and a broad range of applications.

## Features
- Facial recognition
- User management
- User-friendly interface
- Open source
- Compatible with any intercom system with open video streaming
- Telegram bot
- Web interface

## Work stages
### Find video stream from my house's intercom
![](https://stcdn.business-online.ru/articles/3f/1623946702_BO_ELG_3551.jpg)

We have a video intercom system by Tattelekom in our houses. A system is connected to the Internet and has a mobile application. 

First of all, we need to find a video stream from this intercom. We found the link to the video stream using sniffing tools. 

https://streamer109.tattelecom.ru/intercom_3497/mpegts?token=...

The link contains the number of the intercom and the token. The token is valid for at least 3 months.

So, we have a video stream, and we can start working on the project.

### Face detection
We have used many models, but YOLOv8 was the best in terms of time and accuracy.

### Face recognition
We need to recognize faces in the video stream. We use the DeepFace library for this, VGG-Face model.

### Web interface
We have created a web interface for managing users and add new users to the database. We have used the Streamlit framework for this.

## How to use
### Install dependencies
```bash
pip install -r requirements.txt
```

### Run 
```bash
bash start.sh
```

## Possible Improvements
- [ ] Add a database for storing users
- [ ] Add a Telegram bot
- [ ] Add a web interface for viewing the video stream
- [ ] Add a web interface for viewing the logs
- [ ] Select and fine-tune the face recognition model for better accuracy
