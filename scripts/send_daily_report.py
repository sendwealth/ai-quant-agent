#!/usr/bin/env python3
"""
发送日常状态报告邮件
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import os

def load_env():
    """加载环境变量"""
    env_file = Path(__file__).parent.parent / '.env'
    env_vars = {}
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # 移除注释
                if '#' in value:
                    value = value.split('#')[0].strip()
                env_vars[key.strip()] = value.strip()
    
    return env_vars

def send_report_email():
    """发送报告邮件"""
    # 加载配置
    env = load_env()
    
    # 读取报告内容
    report_file = Path(__file__).parent.parent / 'reports' / f'daily_status_{datetime.now().strftime("%Y%m%d")}.md'
    
    if not report_file.exists():
        print(f"❌ 报告文件不存在: {report_file}")
        return False
    
    with open(report_file, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    # 创建邮件
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'[AI量化系统] 日常状态报告 - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    msg['From'] = env['EMAIL_SENDER']
    msg['To'] = env['EMAIL_RECIPIENTS']
    
    # 纯文本版本
    text_part = MIMEText(report_content, 'plain', 'utf-8')
    msg.attach(text_part)
    
    # 发送邮件
    try:
        with smtplib.SMTP_SSL(env['EMAIL_SMTP_SERVER'], int(env['EMAIL_SMTP_PORT'])) as server:
            server.login(env['EMAIL_SENDER'], env['EMAIL_PASSWORD'])
            server.sendmail(
                env['EMAIL_SENDER'],
                env['EMAIL_RECIPIENTS'].split(','),
                msg.as_string()
            )
        
        print(f"✅ 邮件发送成功")
        print(f"   收件人: {env['EMAIL_RECIPIENTS']}")
        print(f"   主题: {msg['Subject']}")
        return True
    
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False

if __name__ == '__main__':
    send_report_email()
