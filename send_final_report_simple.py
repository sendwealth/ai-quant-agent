#!/usr/bin/env python3
"""
发送最终模拟盘报告邮件 - 简化版
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

def read_env_file():
    """直接读取.env文件"""
    env_path = Path('.env')
    config = {}
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    return config

def send_email_report():
    """发送邮件报告"""
    
    # 读取配置
    config = read_env_file()
    
    smtp_server = config.get('EMAIL_SMTP_SERVER', 'smtp.163.com')
    smtp_port = int(config.get('EMAIL_SMTP_PORT', 465))
    sender_email = config.get('EMAIL_SENDER', '')
    sender_password = config.get('EMAIL_PASSWORD', '')
    receiver_email = config.get('EMAIL_RECIPIENTS', 'sendwealth@163.com')
    
    print(f"SMTP服务器: {smtp_server}:{smtp_port}")
    print(f"发送者: {sender_email}")
    print(f"收件人: {receiver_email}")
    print(f"授权码状态: {'已配置' if sender_password else '未配置'}")
    
    if not sender_password:
        print("❌ 邮箱授权码未配置")
        return False
    
    # 读取最终报告
    report_path = Path('final_simulation_report.md')
    if not report_path.exists():
        print("❌ 最终报告文件不存在")
        return False
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    print(f"📄 报告文件大小: {len(report_content)} 字符")
    
    # 创建邮件
    msg = MIMEMultipart()
    msg['From'] = f'AI Quant System <{sender_email}>'
    msg['To'] = receiver_email
    msg['Subject'] = f'AI Quant System Final Simulation Report - {datetime.now().strftime("%Y-%m-%d")}'
    
    # 邮件内容
    email_body = f"""AI Quant Trading System - Final Simulation Report

Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Recipient: {receiver_email}
Report Type: Simulation Operation Summary

Full Report Content:

{report_content}

---
This email was automatically sent by AI Quant Trading System
For technical issues, please contact system administrator
"""
    
    msg.attach(MIMEText(email_body, 'plain', 'utf-8'))
    
    try:
        print("📧 正在发送邮件...")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print("✅ 最终报告邮件发送成功！")
        print(f"   收件人: {receiver_email}")
        print("   请检查收件箱（或垃圾邮件文件夹）")
        return True
        
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False

if __name__ == '__main__':
    success = send_email_report()
    if success:
        print("\n🎉 任务完成!")
        print("✅ ai-quant-agent项目状态检查完成")
        print("✅ 模拟盘操作执行完成") 
        print("✅ 最终报告已发送至指定邮箱")
    else:
        print("\n❌ 任务失败")