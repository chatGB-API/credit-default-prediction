@echo off
chcp 65001
title 信用违约预测 - Streamlit

echo ================================
echo   正在启动信用违约预测系统...
echo ================================
echo.

call conda activate ai_llm
cd /d D:\study\compute\github\credit-default-prediction\src
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false


streamlit run streamlit_app.py

echo.
echo 程序已退出
pause