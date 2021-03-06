# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

from collections import Counter
import numpy as np
from random import randint
import json
import random

from message import message
from config import global_config as cfg
import time


class user():
    def __init__(self, user_id, busi_id):
        self.user_id = user_id
        self.busi_id = busi_id
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]

    def find_brother(self, node):
        return [child.name for child in node.parent.children if child.name != node.name]

    def find_children(self, node):
        return [child.name for child in node.children if child.name != node.name]

    def inform_facet(self, facet):
        data = dict()
        data['facet'] = facet

        if facet not in cfg.item_dict[str(self.busi_id)]['categories']:
            data['value'] = None
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
        else:
            data['value'] = [facet]
            return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

    def response(self, input_message):
        assert input_message.sender == cfg.AGENT
        assert input_message.receiver == cfg.USER

        # _______ update candidate _______
        if 'candidate' in input_message.data: self.recent_candidate_list = input_message.data['candidate']

        new_message = None
        if input_message.message_type == cfg.EPISODE_START or input_message.message_type == cfg.ASK_FACET:
            facet = input_message.data['facet']
            new_message = self.inform_facet(facet)

        if input_message.message_type == cfg.MAKE_REC:
            if self.busi_id in input_message.data['rec_list']:
                data = dict()
                data['ranking'] = input_message.data['rec_list'].index(self.busi_id) + 1
                data['total'] = len(input_message.data['rec_list'])
                new_message = message(cfg.USER, cfg.AGENT, cfg.ACCEPT_REC, data)
            else:
                data = dict()
                data['rejected_item_list'] = input_message.data['rec_list']
                new_message = message(cfg.USER, cfg.AGENT, cfg.REJECT_REC, data)
        return new_message
    # end def
