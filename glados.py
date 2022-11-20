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
subparsers = parser.add_subparsers(dest='action', help='sub-command help')

# create the parser for the "a" command
sayParser = subparsers.add_parser(
    'say', 
    allow_abbrev=True,
    help='Generates spoken audio from text and outputs it to an audio device'
)
sayParser.add_argument('--input', '-i')
sayParser.add_argument('text', nargs='*')

generateParser = subparsers.add_parser(
    'generate',
    aliases=['gen'],
    allow_abbrev=True
)
generateParser.add_argument('--input', '-i')
generateParser.add_argument('--output', '-o', required=True)
generateParser.add_argument('text', nargs='*')

# create the parser for the "b" command
cuiParser = subparsers.add_parser('cui', help='Runs the TTS engine in an interactive CUI mode.')
# parser_b.add_argument("--opt3", action='store_true')
# parser_b.add_argument("--opt4", action='store_true')

# create the parser for the "b" command
serverParser = subparsers.add_parser('serve', help='Runs the TTS engine in a server mode.')
serverParser.add_argument("--port", help='the port to run the server on', default=5000)
serverParser.add_argument("--host", help='the host to run the server on', default="127.0.0.1")

# parser.add_argument('action', help = 'Action what should happen', choices = ['cli', 'cui', 'server'])

# args, unknownargs = parser.parse_known_args(['serve','--port', '1234'])
# args, unknownargs = parser.parse_known_args(['say', '-i', 'input.txt' , "Hello world", "world hello!"])
# args, unknownargs = parser.parse_known_args(['say',"Hello world","Hello world1"])
# args, unknownargs = parser.parse_known_args(['gen', '-o', 'output.wav', "Hello world", "Hello world1"])
args, unknownargs = parser.parse_known_args()
# args = parser.parse_args(['server'])

if args.action is None:
    exit()

engine = TTSEngine.TTSEngine()
engine.warmup()
# engine.generate("It shouldn't be hard to stay alive long enough to find him.").play()
# engine.generate("I thought of some good news. He's going to run out of test chambers eventually. I never stockpiled them.").play()
# exit()

print("Action: " + args.action)


if args.action == 'serve':
    print("Port: " + args.port)
    server = TTSServer.TTSServer(engine, args.host, args.port)
    server.start()

elif args.action == 'say':
    sayText = tools.getInputText(args)
    print("Say: " + sayText)
    engine.generate(sayText).play()

elif args.action == 'generate' or args.action == 'gen':
    if args.output is None:
        print("Error: No output file specified")
        exit()

    sayText = tools.getInputText(args)
    print("Generate: " + sayText)
    engine.generate(sayText).save(args.output)

elif args.action == 'cui':
    while(1):
        text = input("Input: ")
        result = engine.generate(text)
        print(result)
        result.play()





# GlaDos say: Generates a glados voice audio from the given text and plays it back.
# GlaDos gen: Generates a glados voice audio from the given text and saves it to the given file.
# -cli: Runs the TTS engine in CLI mode. One command generates one audio file. This is the default mode.
# -cui: Runs the TTS engine in an interactive CUI mode.
# GlaDos serve: Starts a server that can be used to generate audio files.
