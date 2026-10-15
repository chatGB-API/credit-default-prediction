@echo off
chcp 65001 > nul

cd /d D:\study\compute\github\credit-default-prediction\src

D:\anaconda\envs\ai_llm\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

pause