@echo off
title 纸月 Zhiyue-Bot 启动器 (Server优先模式)

echo [1/3] 正在清理残留进程，确保端口 18001 释放...
taskkill /f /im python.exe /t >nul 2>&1
taskkill /f /im NapCat.exe /t >nul 2>&1
timeout /t 1 >nul

echo [2/3] 正在启动“纸月大脑” (WebSocket Server)...
:: 请确保这里的路径是你纸月的实际路径
cd /d "D:\Github_Storage\cyber_daughter_relate\Zhiyue-Bot"
start "Zhiyue-Core" cmd /k "call .\venv\Scripts\activate && python main.py"

echo.
echo -------------------------------------------------------
echo 正在等待大脑初始化并开启 18001 端口...
echo (通常需要 5-10 秒，等待加载 LLM 和 向量数据库)
echo -------------------------------------------------------
:: 这里设置 8 秒等待时间，确保纸月已经显示 "server listening"
timeout /t 8

echo [3/3] 正在启动“NapCat身体” (WebSocket Client)...
:: 请确保这里的路径是你 NapCat 的实际路径
cd /d "C:\0_Storage\qqbot相关\NapCat.Shell.Windows.OneKey\NapCat.44498.Shell"
start "NapCat-Shell" napcat.bat

echo.
echo [完成] 启动序列执行完毕！请检查纸月窗口是否显示 "OneBot connected"。
pause