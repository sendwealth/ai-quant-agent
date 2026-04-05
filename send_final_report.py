#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发送最终模拟盘报告邮件
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import json
import os
from pathlib import Path
from datetime import datetime

def send_final_report():
    """发送最终模拟盘报告"""
    
    # 读取.env文件
    env_path = Path('.env')
    smtp_server = 'smtp.163.com'
    smtp_port = 465
    sender_email = 'sendwealth@163.com'
    sender_password = ''
    receiver_email = 'sendwealth@163.com'
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'EMAIL_SMTP_SERVER':
                        smtp_server = value
                    elif key == 'EMAIL_SMTP_PORT':
                        smtp_port = int(value)
                    elif key == 'EMAIL_SENDER':
                        sender_email = value
                    elif key == 'EMAIL_PASSWORD':
                        sender_password = value
                    elif key == 'EMAIL_RECIPIENTS':
                        receiver_email = value
    
    if not sender_password:
        print("❌ 邮箱授权码未配置")
        return False
    
    if not sender_password:
        print("❌ 邮箱授权码未配置")
        return False
    
    # 读取最终报告
    report_path = Path('final_simulation_report.md')
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
    else:
        print("❌ 最终报告文件不存在")
        return False
    
    # 创建邮件
    msg = MIMEMultipart()
    msg['From'] = f'AI量化交易系统 <{sender_email}>'
    msg['To'] = receiver_email
    msg['Subject'] = Header(f'AI量化交易系统模拟盘操作最终报告 - {datetime.now().strftime("%Y-%m-%d")}', 'utf-8')
    
    # 添加邮件正文
    msg.attach(MIMEText(report_content, 'plain', 'utf-8'))
    
    try:
        # 发送邮件
        print(f"📧 正在发送邮件到 {receiver_email}...")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, sender_password)
        
        # 手动编码邮件内容
        raw_message = msg.as_bytes()
        server.send_message(msg)
        server.quit()
        print("✅ 最终报告邮件发送成功！")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False

if __name__ == '__main__':
    send_final_report()