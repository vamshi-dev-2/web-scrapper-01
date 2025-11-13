--WEB SCRAPER--
The project is about taking a URL and search query to take all the webpages' content that contains the search query and to display it using Single Page Applications

Technologies Used:
-->Django for Backend
-->ReactJS for Frontend
-->Milvus for vector storing and Indexing

FEATURES:
-->uses SPA, which makes the UI seamless
-->The accurate results with Milvus models
-->Shorter run time and accurate results

DEMO
<img width="1910" height="792" alt="image" src="https://github.com/user-attachments/assets/89ee4cc1-de1a-4d89-95fc-43b0f1f59009" />

<img width="1507" height="777" alt="image" src="https://github.com/user-attachments/assets/d0a1c783-94e6-468c-a94f-5347c0f607a7" />


INSTALLATION:
-->Clone the project
>> git clone https://github.com/vamshi-dev-2/web-scrapper-01.git

MAKE SURE ALL DEPENDENCIES ARE INSTALLED
Make sure Docker is running and Milvus is active. You can make sure by command >> docker ps 
"python manage.py makemigrations"


RUNNING THE APP

demo/
>> python manage.py runserver

frontend/
>> npm start


FOLDER STRUCTURE
project-root/
│
├── demo/                     # Django project folder (main backend project)
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── searchapp/                # Django app containing indexing + search APIs
│   ├── migrations/
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
│
├── frontend/                 # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   │   ├── ResultCard.js
│   │   │   └── ResultCard.css
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   ├── logo.svg
│   │   ├── reportWebVitals.js
│   │   └── setupTests.js
│   ├── package.json
│   └── package-lock.json
│
├── docker-compose.yml        # container orchestration
├── db.sqlite3                # SQLite database (local dev)
├── manage.py                 # Django entry point
└── README.md                 # Project documentation



