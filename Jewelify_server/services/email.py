# services/email.py
import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

# Configure logging based on environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logging_level = logging.DEBUG if ENVIRONMENT == "development" else logging.WARNING
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SMTP configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Jewelify")


def send_otp_email(to_email: str, otp: str, otp_expiry_minutes: int = 5) -> bool:
    """Send an OTP verification email to the specified address.

    Returns True on success, False if credentials are missing or SMTP fails.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.error(
            "SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD environment variables."
        )
        return False

    subject = "Your Jewelify verification code"
    from_address = f"{SMTP_FROM_NAME} <{SMTP_USER}>"

    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Jewelify OTP</title>
</head>
<body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f4;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="520" cellpadding="0" cellspacing="0"
               style="background-color:#ffffff;border-radius:8px;overflow:hidden;
                      box-shadow:0 2px 8px rgba(0,0,0,0.08);">
          <!-- Header -->
          <tr>
            <td align="center" style="background-color:#1a1a2e;padding:32px 40px;">
              <h1 style="margin:0;color:#e8c97e;font-size:28px;letter-spacing:2px;">Jewelify</h1>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="padding:40px 40px 32px;">
              <p style="margin:0 0 16px;color:#333333;font-size:16px;">Hello,</p>
              <p style="margin:0 0 24px;color:#555555;font-size:15px;line-height:1.6;">
                Use the verification code below to complete your request.
                This code is valid for <strong>{otp_expiry_minutes} minutes</strong>.
              </p>
              <!-- OTP box -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding:24px 0;">
                    <div style="display:inline-block;background-color:#f0f0f0;border-radius:6px;
                                padding:16px 40px;font-size:36px;font-weight:bold;
                                letter-spacing:10px;color:#1a1a2e;">
                      {otp}
                    </div>
                  </td>
                </tr>
              </table>
              <p style="margin:24px 0 0;color:#888888;font-size:13px;line-height:1.5;">
                If you did not request this code, you can safely ignore this email.
                Do not share this code with anyone.
              </p>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="background-color:#f9f9f9;padding:20px 40px;border-top:1px solid #eeeeee;">
              <p style="margin:0;color:#aaaaaa;font-size:12px;text-align:center;">
                &copy; 2024 Jewelify. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    text_body = (
        f"Your Jewelify verification code is: {otp}\n"
        f"This code is valid for {otp_expiry_minutes} minutes.\n"
        "Do not share this code with anyone."
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        logger.info(f"OTP email sent successfully to {to_email}")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error(
            "SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD."
        )
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error while sending OTP email to {to_email}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while sending OTP email to {to_email}: {str(e)}")
        return False
