import os
import argparse
import hashlib

import sys
import os
import torch
import utils.tools as tools
from scipy.io.wavfile import write
import time
import hashlib
import utils.ttsEngine as TTSEngine
import utils.ttsServer as TTSServer
import wave

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

# Contains multiple actions: ['cli', 'cui', 'server']
subparsers = parser.add_subparsers(dest='action',help='sub-command help')

# create the parser for the "a" command
parser_cli = subparsers.add_parser('cli', help='Runs the TTS engine in CLI mode. One command generates one audio file. This is the default mode.')
# parser_cli.add_argument("--noAudio", action='store_true')
# parser_cli.add_argument("--opt2", action='store_true')

# create the parser for the "b" command
parser_b = subparsers.add_parser('cui', help='Runs the TTS engine in an interactive CUI mode.')
# parser_b.add_argument("--opt3", action='store_true')
# parser_b.add_argument("--opt4", action='store_true')

# create the parser for the "b" command
parser_b = subparsers.add_parser('server', help='b help')
parser_b.add_argument("--port", help='the port to run the server on', default=5000)

# parser.add_argument('action', help = 'Action what should happen', choices = ['cli', 'cui', 'server'])

args = parser.parse_args(['server','--port', '1234'])
# args = parser.parse_args(['server'])

engine = TTSEngine.TTSEngine()
engine.warmup()
print("Action: " + args.action)
if args.action == 'server':
    print("Port: " + args.port)
    server = TTSServer.TTSServer(engine, args.port)
    server.start()

elif args.action == 'cli':
    print("cli")
elif args.action == 'cui':
    print("cui")
