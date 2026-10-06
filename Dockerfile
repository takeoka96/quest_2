FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# GA4トラッキングコードをStreamlitのindex.htmlに注入(GC GA4)
RUN python -c "import streamlit, os; path = os.path.join(os.path.dirname(streamlit.__file__), 'static', 'index.html'); content = open(path).read(); tag = '<script async src=\"https://www.googletagmanager.com/gtag/js?id=G-RR85JMV8J8\"></script><script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag(\"js\",new Date());gtag(\"config\",\"G-RR85JMV8J8\");</script>'; open(path, 'w').write(content.replace('</head>', tag + '</head>', 1))"
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
