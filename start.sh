#!/bin/bash

echo "========================================"
echo "电力设备监控智能问答系统 - 启动脚本"
echo "========================================"
echo ""

echo "[1/2] 启动FastAPI后端服务..."
python app.py &
BACKEND_PID=$!
echo "后端服务已启动 (PID: $BACKEND_PID)"

sleep 3

echo "[2/2] 启动前端HTTP服务器..."
python -m http.server 3000 &
FRONTEND_PID=$!
echo "前端服务已启动 (PID: $FRONTEND_PID)"

echo ""
echo "========================================"
echo "服务启动完成！"
echo ""
echo "后端服务: http://localhost:8000"
echo "API文档:   http://localhost:8000/docs"
echo "前端界面: http://localhost:3000"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "========================================"

trap "echo ''; echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo '服务已停止'; exit" INT TERM

wait
