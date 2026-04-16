@echo off
chcp 65001 >nul
echo ========================================
echo 电力设备监控智能问答系统 - 启动脚本
echo ========================================
echo.

echo [1/2] 启动FastAPI后端服务...
start "FastAPI后端" cmd /k "python app.py"

timeout /t 3 /nobreak >nul

echo [2/2] 启动前端HTTP服务器...
start "前端界面" cmd /k "python -m http.server 3000"

echo.
echo ========================================
echo 服务启动完成！
echo.
echo 后端服务: http://localhost:8000
echo API文档:   http://localhost:8000/docs
echo 前端界面: http://localhost:3000
echo.
echo 按任意键关闭此窗口（服务将继续运行）
echo ========================================
pause >nul