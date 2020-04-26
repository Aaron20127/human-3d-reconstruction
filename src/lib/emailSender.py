import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import os


class emailSender(object):
    def __init__(self, host, user, pwd, sender, receiver):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.sender = sender
        self.receiver = receiver
        self.update_msg()

    def send_email(self, title, text='', file_paths=[]):
        self.title = title
        self.pack_text(text)

        for fp in file_paths:
            self.pack_file(fp)

        self.send()

    def send(self):
        """ send message.
        :param title: mail title.
        :param msg: mail content.
        :return:
        """
        self.msg['From'] = "{}".format(self.sender)
        self.msg['To'] = ",".join(self.receiver)
        self.msg['Subject'] = self.title

        try:
            smtpObj = smtplib.SMTP_SSL(self.host, 465)  # 启用SSL发信, 端口一般是465
            smtpObj.login(self.user, self.pwd)  # 登录验证
            smtpObj.sendmail(self.sender, self.receiver, self.msg.as_string())  # 发送
            print("Send mail to {} successfully.".format(self.receiver))
        except smtplib.SMTPException as e:
            print(e)

        self.update_msg()

    def update_msg(self):
        self.msg = MIMEMultipart('mixed')

    def pack_text(self, text):
        """ package string message """
        text = MIMEText(text, 'plain', 'utf-8')
        self.msg.attach(text)

    def pack_file(self, file_path):
        """ package file """
        fr = open(file_path, 'rb').read()
        text = MIMEText(fr, 'base64', 'utf-8')
        text["Content-Type"] = 'application/octet-stream'

        _, file_name = os.path.split(file_path)
        text["Content-Disposition"] = 'attachment; filename="{}"'.format(file_name) ## 获取文件名
        self.msg.attach(text)


def send_email(opt):
    """send email when complete training. """
    sender = emailSender(opt.mail_host, opt.mail_user, opt.mail_pwd,
                         opt.mail_sender, opt.mail_receiver)

    title = opt.title
    text = 'train over.'
    file_paths = [opt.save_dir + '/opt.txt',
                  opt.save_dir + '/logs/train.txt',
                  opt.save_dir + '/logs/val.txt']

    sender.send_email(title, text, file_paths)


if __name__ == '__main__':
    mail_host = "smtp.163.com"  # SMTP服务器
    mail_user = "lwalgorithm@163.com"  # 用户名
    mail_pwd = "RZVYBFVFKEASVHQO"  # 授权密码，非登录密码

    mail_sender = 'lwalgorithm@163.com'  # 发件人邮箱(最好写全, 不然会失败)
    mail_receivers = 'lwalgorithm@163.com'

    sender = emailSender(mail_host, mail_user, mail_pwd,
                             mail_sender, mail_receivers)

    ## message
    title = '人生苦短 go'  # 邮件主题
    text = '我用Python,'

    ## send email
    sender.send_email(title, text)
