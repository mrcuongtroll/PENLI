import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import itertools
from utils.utils import AverageMeter, compute_masked_lm_results, compute_generative_results
from utils.config import ConfigParser
from model.model import BertPENLI, T5PENLI
from model.rl.actor_critic import A2C
from definitions import *

logger = logging.getLogger(name=__name__)


# Classes
class RLTrainer:

    def __init__(self,
                 module: A2C,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 valid_loader: DataLoader = None,
                 lr_scheduler=None,
                 config: ConfigParser = None,
                 device='cuda'
                 ):
        self.device = device
        self.module = module.to(device)
        self.critic_loss_fn = nn.MSELoss()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.current_epoch = 0
        self.current_batch_idx = 0
        self.best_reward = 0
        if self.config['rl']['freeze_plm']:
            self.module.model.freeze_plm(freeze=True)

    def train(self, resume=False):
        with torch.autograd.set_detect_anomaly(True):
            if resume:
                self._load_model()
                logger.info(f"------> Resume training from epoch {self.current_epoch}, "
                            f"batch_idx {self.current_batch_idx}...")
            else:
                logger.info(f"------> Begin training...")
            for epoch in range(self.current_epoch, self.config['rl']['num_epochs']):
                logger.info(f"------> Current epoch: {epoch}")
                train_result = self._train_epoch(epoch)

                if self.valid_loader is not None:
                    valid_result = self._eval_epoch(epoch)
                    valid_loss = valid_result['loss']
                    valid_reward = valid_loss['reward']
                    valid_result['state_dict'] = self.module.model.state_dict()
                    valid_result['critic_state_dict'] = self.module.critic_head.state_dict()
                    valid_result['optimizer'] = self.optimizer.state_dict()
                    valid_result['lr_scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler else None
                    valid_result['epoch'] = epoch + 1
                    valid_result['batch_idx'] = 0
                    valid_result['train_sampler'] = self.train_loader.sampler.get_state()
                    if valid_reward > self.best_reward:
                        self.best_reward = valid_reward
                        logger.info(f"New best valid reward found: {valid_reward}. Saving checkpoint...")
                        valid_result['best_reward'] = self.best_reward
                        torch.save(valid_result, os.path.join(self.config['save_dir'], 'rl', 'checkpoint_best.pt'))
                    logger.info(f"------> Saving checkpoint...")
                    valid_result['best_reward'] = self.best_reward
                    torch.save(valid_result, os.path.join(self.config['save_dir'], 'rl', 'checkpoint_last.pt'))
                    logger.info("------> Done.")
        return

    def _train_epoch(self, epoch):
        result = None
        self.module.train()
        reward_meter = AverageMeter()
        loss_meter = AverageMeter()
        num_batches_per_print = len(self.train_loader) // self.config['rl']['num_prints']
        for batch_idx, batch in enumerate(itertools.islice(self.train_loader, self.current_batch_idx, None)):
            rewards, critic_vals, action_lp_vals, entropy, total_rewards = self.module.train_env_episode(batch)
            loss = self.module.compute_loss(action_p_vals=action_lp_vals,
                                            G=rewards,
                                            V=critic_vals,
                                            entropy=entropy,
                                            critic_loss_fn=self.critic_loss_fn)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config['rl']['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(self.module.parameters(), self.config['rl']['grad_clip'])
            self.optimizer.step()
            lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler:
                self.lr_scheduler.step()
            loss_meter.update(loss.item())
            reward_meter.update(total_rewards.mean().item())
            result = {"loss": loss_meter.average(),
                      "reward": reward_meter.average(),
                      "lr": lr}
            if (batch_idx + 1) % num_batches_per_print == 0:
                logger.info(f"Training "
                            f"[{batch_idx * self.train_loader.batch_size}/{len(self.train_loader.dataset)} "
                            f"({(100. * batch_idx / len(self.train_loader)):.0f}%)] "
                            f"| Loss: {loss_meter.average():.5f} "
                            f"| Reward: {reward_meter.average():.4f}"
                            f"| Learning Rate: {lr}")
                # Save checkpoint
                logger.info(f"------> Saving checkpoint...")
                result['state_dict'] = self.module.model.state_dict()
                result['critic_state_dict'] = self.module.critic_head.state_dict()
                result['optimizer'] = self.optimizer.state_dict()
                result['lr_scheduler'] = self.lr_scheduler.state_dict() if self.lr_scheduler else None
                result['epoch'] = epoch
                result['batch_idx'] = self.current_batch_idx + batch_idx + 1
                result['best_reward'] = self.best_reward
                result['train_sampler'] = self.train_loader.sampler.get_state()
                torch.save(result, os.path.join(self.config['save_dir'], 'rl', 'checkpoint_last.pt'))
                logger.info("------> Done.")
        logger.info(f"Finished training epoch {epoch} "
                    f"| Loss: {loss_meter.average():.5f} "
                    f"| Reward: {reward_meter.average():.4f}"
                    f"| Learning Rate: {self.optimizer.param_groups[0]['lr']}"
                    )
        return result

    def _eval_epoch(self, epoch):
        self.module.eval()
        result = None
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        num_batches_per_print = len(self.valid_loader) // self.config['rl']['num_prints']
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                rewards, critic_vals, action_lp_vals, entropy, total_rewards = self.module.train_env_episode(batch)
                loss = self.module.compute_loss(action_p_vals=action_lp_vals,
                                                G=rewards,
                                                V=critic_vals,
                                                entropy=entropy,
                                                critic_loss_fn=self.critic_loss_fn)
                loss_meter.update(loss.item())
                reward_meter.update(total_rewards.mean())
                result = {"loss": loss_meter.average(),
                          "reward": reward_meter.average()
                          }
                if (batch_idx + 1) % num_batches_per_print == 0:
                    logger.info(f"Evaluating "
                                f"[{batch_idx * self.valid_loader.batch_size}/{len(self.valid_loader.dataset)} "
                                f"({(100. * batch_idx / len(self.valid_loader)):.0f}%)] "
                                f"| Loss: {loss_meter.average():.5f} "
                                f"| Reward: {reward_meter.average():.4f}"
                                )
        logger.info(f"Finished training epoch {epoch} "
                    f"| Loss: {loss_meter.average():.5f} "
                    f"| Reward: {reward_meter.average():.4f}"
                    )
        return result

    def _load_model(self):
        checkpoint_path = os.path.join(self.config['save_dir'], 'rl', 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.module.model.load_state_dict(checkpoint['state_dict'])
        self.module.critic_head.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.param_groups[0]['capturable'] = True
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.best_reward = checkpoint['best_reward']
        self.current_epoch = checkpoint['epoch']
        self.current_batch_idx = checkpoint['batch_idx']
        self.train_loader.sampler.set_state(checkpoint['train_sampler'])
        logger.info(f"------> Loaded checkpoint from {checkpoint_path}")
