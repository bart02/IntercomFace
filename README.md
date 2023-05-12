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

### Face recognition
We need to recognize faces in the video stream. We use the OpenCV library for this. We use the Haar cascade classifier to detect faces. We use the LBPH (Local Binary Patterns Histograms) algorithm to recognize faces.