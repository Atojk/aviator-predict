Deployment Instructions for Render.com:

1. Push this folder to a GitHub repo:
   git init
   git add .
   git commit -m "Initial"
   git remote add origin https://github.com/yourname/aviator-predict.git
   git push -u origin master

2. Go to https://render.com
   - Create new Web Service
   - Connect to your repo
   - Choose Python
   - Start command: gunicorn app:app

3. After it's live at https://aviator-predict.onrender.com:
   - Go to your eu5.org domain settings
   - Add CNAME record:
     Host: aviator-predict.eu5.org
     Target: aviator-predict.onrender.com
   - Wait for DNS to propagate (30 min–4 hrs)

4. Done!