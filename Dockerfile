FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# GA4トラッキングコードをStreamlitのindex.htmlに注入(GC GA4)
RUN sed -i "s|</head>|<script async src='https://www.googletagmanager.com/gtag/js?id=G-RR85JMV8J8'></script><script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','G-RR85JMV8J8');</script></head>|" $(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static', 'index.html'))")

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
