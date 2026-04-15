from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random


class PowerEquipmentTools:
    """电力设备监控专用工具集"""
    
    def __init__(self):
        """初始化工具集，模拟一些设备数据"""
        self.equipment_status = {
            "变压器#1": {"status": "正常", "temperature": 45, "oil_level": "正常", "last_check": "2026-04-14"},
            "变压器#2": {"status": "告警", "temperature": 85, "oil_level": "偏低", "last_check": "2026-04-15"},
            "断路器#1": {"status": "正常", "position": "合闸", "last_check": "2026-04-14"},
            "电容器组#1": {"status": "正常", "voltage": 10.5, "current": 50, "last_check": "2026-04-13"},
            "母线#1": {"status": "正常", "voltage": 110, "frequency": 50, "last_check": "2026-04-15"}
        }
        
        self.alerts = [
            {"id": 1, "equipment": "变压器#2", "type": "温度过高", "level": "严重", "time": "2026-04-15 08:30:00", "status": "未处理"},
            {"id": 2, "equipment": "电容器组#1", "type": "电压波动", "level": "警告", "time": "2026-04-14 15:20:00", "status": "已处理"},
            {"id": 3, "equipment": "母线#1", "type": "频率异常", "level": "信息", "time": "2026-04-13 10:15:00", "status": "已处理"}
        ]
        
        self.maintenance_plans = [
            {"id": 1, "equipment": "变压器#1", "type": "预防性试验", "date": "2026-05-01", "status": "待执行", "description": "变压器油色谱分析、绝缘电阻测试"},
            {"id": 2, "equipment": "断路器#1", "type": "机构检修", "date": "2026-04-20", "status": "执行中", "description": "断路器机械特性测试、触头检查"},
            {"id": 3, "equipment": "电容器组#1", "type": "日常维护", "date": "2026-04-10", "status": "已完成", "description": "外观检查、接头温度检测"}
        ]
    
    def check_equipment_status(self, equipment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        检查设备状态
        
        Args:
            equipment_name: 设备名称，如果为None则返回所有设备状态
            
        Returns:
            设备状态信息
        """
        if equipment_name:
            if equipment_name in self.equipment_status:
                return {
                    "success": True,
                    "data": {equipment_name: self.equipment_status[equipment_name]},
                    "message": f"成功获取设备 {equipment_name} 的状态信息"
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "message": f"设备 {equipment_name} 不存在"
                }
        else:
            return {
                "success": True,
                "data": self.equipment_status,
                "message": "成功获取所有设备状态信息"
            }
    
    def query_alerts(self, status: Optional[str] = None, level: Optional[str] = None) -> Dict[str, Any]:
        """
        查询告警信息
        
        Args:
            status: 告警状态（未处理、已处理、全部）
            level: 告警级别（信息、警告、严重、全部）
            
        Returns:
            告警信息列表
        """
        filtered_alerts = self.alerts
        
        if status and status != "全部":
            filtered_alerts = [alert for alert in filtered_alerts if alert["status"] == status]
        
        if level and level != "全部":
            filtered_alerts = [alert for alert in filtered_alerts if alert["level"] == level]
        
        return {
            "success": True,
            "data": filtered_alerts,
            "count": len(filtered_alerts),
            "message": f"成功查询到 {len(filtered_alerts)} 条告警信息"
        }
    
    def get_maintenance_plans(self, status: Optional[str] = None) -> Dict[str, Any]:
        """
        获取维护计划
        
        Args:
            status: 计划状态（待执行、执行中、已完成、全部）
            
        Returns:
            维护计划列表
        """
        filtered_plans = self.maintenance_plans
        
        if status and status != "全部":
            filtered_plans = [plan for plan in filtered_plans if plan["status"] == status]
        
        return {
            "success": True,
            "data": filtered_plans,
            "count": len(filtered_plans),
            "message": f"成功获取 {len(filtered_plans)} 条维护计划"
        }
    
    def create_maintenance_plan(self, equipment: str, plan_type: str, date: str, description: str) -> Dict[str, Any]:
        """
        创建维护计划
        
        Args:
            equipment: 设备名称
            plan_type: 计划类型
            date: 计划日期
            description: 计划描述
            
        Returns:
            创建结果
        """
        new_id = len(self.maintenance_plans) + 1
        new_plan = {
            "id": new_id,
            "equipment": equipment,
            "type": plan_type,
            "date": date,
            "status": "待执行",
            "description": description
        }
        self.maintenance_plans.append(new_plan)
        
        return {
            "success": True,
            "data": new_plan,
            "message": f"成功创建维护计划，ID: {new_id}"
        }
    
    def acknowledge_alert(self, alert_id: int) -> Dict[str, Any]:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            确认结果
        """
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["status"] = "已处理"
                return {
                    "success": True,
                    "data": alert,
                    "message": f"成功确认告警，ID: {alert_id}"
                }
        
        return {
            "success": False,
            "data": None,
            "message": f"告警 {alert_id} 不存在"
        }


def get_tools() -> List[Dict[str, Any]]:
    """获取工具列表，用于Langchain Agent"""
    from langchain.tools import StructuredTool
    
    tools = PowerEquipmentTools()
    
    # 设备状态检查工具
    check_status_tool = StructuredTool.from_function(
        func=tools.check_equipment_status,
        name="check_equipment_status",
        description="检查电力设备的运行状态，包括温度、油位、开关位置等信息。可以指定具体设备名称或查询所有设备状态。"
    )
    
    # 告警查询工具
    query_alerts_tool = StructuredTool.from_function(
        func=tools.query_alerts,
        name="query_alerts",
        description="查询电力设备的告警信息，支持按状态（未处理/已处理）和级别（信息/警告/严重）筛选。"
    )
    
    # 维护计划管理工具
    get_maintenance_tool = StructuredTool.from_function(
        func=tools.get_maintenance_plans,
        name="get_maintenance_plans",
        description="获取电力设备的维护计划，支持按状态（待执行/执行中/已完成）筛选。"
    )
    
    # 创建维护计划工具
    create_plan_tool = StructuredTool.from_function(
        func=tools.create_maintenance_plan,
        name="create_maintenance_plan",
        description="创建新的设备维护计划，需要提供设备名称、计划类型、日期和描述。"
    )
    
    # 确认告警工具
    acknowledge_alert_tool = StructuredTool.from_function(
        func=tools.acknowledge_alert,
        name="acknowledge_alert",
        description="确认并处理告警信息，需要提供告警ID。"
    )
    
    return [check_status_tool, query_alerts_tool, get_maintenance_tool, create_plan_tool, acknowledge_alert_tool]
