#!/usr/bin/env python3
"""
发送最终模拟盘报告邮件
"""

import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env():
    """加载环境变量"""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    # 处理注释（# 前面的内容）
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()


def read_final_report():
    """读取最终报告文件"""
    report_path = PROJECT_ROOT / 'final_simulation_report.md'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return None


def send_final_report_email():
    """发送最终报告邮件"""

    # 加载环境变量
    load_env()

    # 读取配置
    smtp_server = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.163.com')
    smtp_port = int(os.environ.get('EMAIL_SMTP_PORT', 465))
    sender = os.environ.get('EMAIL_SENDER', '')
    password = os.environ.get('EMAIL_PASSWORD', '')
    recipients_str = os.environ.get('EMAIL_RECIPIENTS', '')
    recipients = [r.strip() for r in recipients_str.split(',') if r.strip()]

    # 检查配置
    print("=" * 60)
    print("邮件配置检查")
    print("=" * 60)
    print(f"SMTP 服务器: {smtp_server}:{smtp_port}")
    print(f"发送者: {sender}")
    print(f"收件人: {', '.join(recipients)}")
    print("=" * 60)

    if not password:
        print("❌ 未配置邮箱授权码")
        return False

    if not recipients:
        print("❌ 未配置收件人")
        return False

    # 读取最终报告
    report_content = read_final_report()
    if not report_content:
        print("❌ 最终报告文件不存在")
        return False

    # 发送报告邮件
    print("\n开始发送最终报告邮件...")

    try:
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"AI量化交易系统模拟盘操作最终报告 - {datetime.now().strftime('%Y-%m-%d')}"

        # 邮件正文
        email_body = f"""
AI量化交易系统 - 模拟盘操作最终报告

{'=' * 60}
执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
收件人: {', '.join(recipients)}
{'=' * 60}

以下是完整的模拟盘操作报告内容：

{report_content}

{'=' * 60}
报告结束
感谢您的关注！
AI量化交易系统
        """

        msg.attach(MIMEText(email_body, 'plain', 'utf-8'))

        # 发送邮件
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(msg)

        print(f"✅ 最终报告邮件发送成功！")
        print(f"   收件人: {', '.join(recipients)}")
        print(f"   报告大小: {len(report_content)} 字符")
        print(f"   请检查收件箱（或垃圾邮件文件夹）")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ 邮箱认证失败")
        print("   可能原因:")
        print("   1. 授权码错误（不是邮箱密码）")
        print("   2. 未开启 SMTP 服务")
        print("   3. 发送者邮箱地址错误")
        return False

    except smtplib.SMTPException as e:
        print(f"❌ SMTP 错误: {e}")
        return False

    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return False


if __name__ == '__main__':
    success = send_final_report_email()
    exit(0 if success else 1)