import wget
import os


if __name__ == '__main__':
    # Download supervised checkpoint
    checkpoint_last_url = 'https://drive.google.com/u/0/uc?id=1Tz8uTPfH1m43Ww50WoIEkmxiH_Sl7x9S&export=download&confirm=t&uuid=8cbfd7b7-3647-4878-a0ee-69e666b54800&at=AKKF8vxp6f9Y-tMiXj8tUuLmv4Pg:1687404629856'
    checkpoint_best_url = 'https://drive.google.com/u/0/uc?id=1gkH6cr61gCIU4xp2abaVAMkBXlDaMw19&export=download&confirm=t&uuid=593573f6-d3e9-438b-be01-7fbe965b9863&at=AKKF8vxz6tAcCmNfhMekx3_ekvYh:1687404593367'
    log_url = 'https://drive.google.com/u/0/uc?id=1IlMqkTiNnQzrZFpZs4fmURQtpbQ1P7Zj&export=download'
    destination_dir = './logs/default_ed'
    destination_dir = os.path.abspath(destination_dir)
    os.makedirs(destination_dir, exist_ok=True)
    wget.download(url=checkpoint_last_url, out=destination_dir)
    wget.download(url=checkpoint_best_url, out=destination_dir)
    wget.download(url=log_url, out=destination_dir)

    # Download critic checkpoint
    critic_checkpoint_last_url = 'https://drive.google.com/u/0/uc?id=1JeoCkQ4vmU92MMwwaDzY4l_Tw16yeWDP&export=download&confirm=t&uuid=e80b87db-6322-4bf6-a9e1-acefcf50626c&at=AKKF8vwpfR0ssjfEofkqFasGwKN1:1687404737894'
    critic_log_url = 'https://drive.google.com/u/0/uc?id=1IlMqkTiNnQzrZFpZs4fmURQtpbQ1P7Zj&export=download'
    critic_destination = './logs/default_ed/critic'
    os.makedirs(critic_destination, exist_ok=True)
    wget.download(url=critic_checkpoint_last_url, out=critic_destination)
    wget.download(url=critic_log_url, out=critic_destination)
