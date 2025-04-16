#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on April 01, 2025, at 04:19
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Cued GoNogo Task'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='H:\\GoNoGo\\Experiment-with-Psychopy\\실습1-Cued GoNogo Task\\Cued GoNogo Task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('Key_Resp') is None:
        # initialise Key_Resp
        Key_Resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Key_Resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "TaskIntro" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='안녕하세요. 검사에 참여해주셔서 감사합니다. 스페이스바를 누르면 검사가 시작됩니다.',
        font='NanumSquare',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Fixation" ---
    FixationPhase = visual.TextStim(win=win, name='FixationPhase',
        text='+',
        font='NanumSquare',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Blank = visual.TextStim(win=win, name='Blank',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Trial" ---
    Cue = visual.Rect(
        win=win, name='Cue',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.5,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    GoNogo_Stim = visual.Rect(
        win=win, name='GoNogo_Stim',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.5,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    Key_Resp = keyboard.Keyboard(deviceName='Key_Resp')
    
    # --- Initialize components for Routine "Feedback" ---
    Feedback_Present = visual.TextStim(win=win, name='Feedback_Present',
        text='',
        font='NanumSquare',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "TaskEnd" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "TaskIntro" ---
    # create an object to store info about Routine TaskIntro
    TaskIntro = data.Routine(
        name='TaskIntro',
        components=[fixation, key_resp],
    )
    TaskIntro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for TaskIntro
    TaskIntro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    TaskIntro.tStart = globalClock.getTime(format='float')
    TaskIntro.status = STARTED
    thisExp.addData('TaskIntro.started', TaskIntro.tStart)
    TaskIntro.maxDuration = None
    # keep track of which components have finished
    TaskIntroComponents = TaskIntro.components
    for thisComponent in TaskIntro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "TaskIntro" ---
    TaskIntro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation* updates
        
        # if fixation is starting this frame...
        if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation.frameNStart = frameN  # exact frame index
            fixation.tStart = t  # local t and not account for scr refresh
            fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation.started')
            # update status
            fixation.status = STARTED
            fixation.setAutoDraw(True)
        
        # if fixation is active this frame...
        if fixation.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            TaskIntro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in TaskIntro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "TaskIntro" ---
    for thisComponent in TaskIntro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for TaskIntro
    TaskIntro.tStop = globalClock.getTime(format='float')
    TaskIntro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('TaskIntro.stopped', TaskIntro.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "TaskIntro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Blocks = data.TrialHandler2(
        name='Blocks',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('BlockList.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(Blocks)  # add the loop to the experiment
    thisBlock = Blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in Blocks:
        currentLoop = Blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # set up handler to look after randomisation of conditions etc
        Trials = data.TrialHandler2(
            name='Trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(Block), 
            seed=None, 
        )
        thisExp.addLoop(Trials)  # add the loop to the experiment
        thisTrial = Trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in Trials:
            currentLoop = Trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "Fixation" ---
            # create an object to store info about Routine Fixation
            Fixation = data.Routine(
                name='Fixation',
                components=[FixationPhase, Blank],
            )
            Fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for Fixation
            Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Fixation.tStart = globalClock.getTime(format='float')
            Fixation.status = STARTED
            thisExp.addData('Fixation.started', Fixation.tStart)
            Fixation.maxDuration = None
            # keep track of which components have finished
            FixationComponents = Fixation.components
            for thisComponent in Fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Fixation" ---
            # if trial has changed, end Routine now
            if isinstance(Trials, data.TrialHandler2) and thisTrial.thisN != Trials.thisTrial.thisN:
                continueRoutine = False
            Fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.2:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *FixationPhase* updates
                
                # if FixationPhase is starting this frame...
                if FixationPhase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    FixationPhase.frameNStart = frameN  # exact frame index
                    FixationPhase.tStart = t  # local t and not account for scr refresh
                    FixationPhase.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(FixationPhase, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixationPhase.started')
                    # update status
                    FixationPhase.status = STARTED
                    FixationPhase.setAutoDraw(True)
                
                # if FixationPhase is active this frame...
                if FixationPhase.status == STARTED:
                    # update params
                    pass
                
                # if FixationPhase is stopping this frame...
                if FixationPhase.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > FixationPhase.tStartRefresh + 0.7-frameTolerance:
                        # keep track of stop time/frame for later
                        FixationPhase.tStop = t  # not accounting for scr refresh
                        FixationPhase.tStopRefresh = tThisFlipGlobal  # on global time
                        FixationPhase.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'FixationPhase.stopped')
                        # update status
                        FixationPhase.status = FINISHED
                        FixationPhase.setAutoDraw(False)
                
                # *Blank* updates
                
                # if Blank is starting this frame...
                if Blank.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
                    # keep track of start time/frame for later
                    Blank.frameNStart = frameN  # exact frame index
                    Blank.tStart = t  # local t and not account for scr refresh
                    Blank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Blank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank.started')
                    # update status
                    Blank.status = STARTED
                    Blank.setAutoDraw(True)
                
                # if Blank is active this frame...
                if Blank.status == STARTED:
                    # update params
                    pass
                
                # if Blank is stopping this frame...
                if Blank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Blank.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        Blank.tStop = t  # not accounting for scr refresh
                        Blank.tStopRefresh = tThisFlipGlobal  # on global time
                        Blank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Blank.stopped')
                        # update status
                        Blank.status = FINISHED
                        Blank.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Fixation" ---
            for thisComponent in Fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Fixation
            Fixation.tStop = globalClock.getTime(format='float')
            Fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Fixation.stopped', Fixation.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Fixation.maxDurationReached:
                routineTimer.addTime(-Fixation.maxDuration)
            elif Fixation.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.200000)
            
            # --- Prepare to start Routine "Trial" ---
            # create an object to store info about Routine Trial
            Trial = data.Routine(
                name='Trial',
                components=[Cue, GoNogo_Stim, Key_Resp],
            )
            Trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            Cue.setSize([Stim_Width, Stim_Height])
            # Run 'Begin Routine' code from ColorSelection
            if StimType == 'Go':
                StimColor = 'green'
            elif StimType == 'NoGo':
                StimColor = 'blue'
            GoNogo_Stim.setFillColor(StimColor)
            GoNogo_Stim.setSize([Stim_Width, Stim_Height])
            # create starting attributes for Key_Resp
            Key_Resp.keys = []
            Key_Resp.rt = []
            _Key_Resp_allKeys = []
            # store start times for Trial
            Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Trial.tStart = globalClock.getTime(format='float')
            Trial.status = STARTED
            thisExp.addData('Trial.started', Trial.tStart)
            Trial.maxDuration = None
            # keep track of which components have finished
            TrialComponents = Trial.components
            for thisComponent in Trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Trial" ---
            # if trial has changed, end Routine now
            if isinstance(Trials, data.TrialHandler2) and thisTrial.thisN != Trials.thisTrial.thisN:
                continueRoutine = False
            Trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Cue* updates
                
                # if Cue is starting this frame...
                if Cue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Cue.frameNStart = frameN  # exact frame index
                    Cue.tStart = t  # local t and not account for scr refresh
                    Cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Cue.started')
                    # update status
                    Cue.status = STARTED
                    Cue.setAutoDraw(True)
                
                # if Cue is active this frame...
                if Cue.status == STARTED:
                    # update params
                    pass
                
                # if Cue is stopping this frame...
                if Cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Cue.tStartRefresh + CueTime-frameTolerance:
                        # keep track of stop time/frame for later
                        Cue.tStop = t  # not accounting for scr refresh
                        Cue.tStopRefresh = tThisFlipGlobal  # on global time
                        Cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Cue.stopped')
                        # update status
                        Cue.status = FINISHED
                        Cue.setAutoDraw(False)
                
                # *GoNogo_Stim* updates
                
                # if GoNogo_Stim is starting this frame...
                if GoNogo_Stim.status == NOT_STARTED and tThisFlip >= CueTime+0.2-frameTolerance:
                    # keep track of start time/frame for later
                    GoNogo_Stim.frameNStart = frameN  # exact frame index
                    GoNogo_Stim.tStart = t  # local t and not account for scr refresh
                    GoNogo_Stim.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(GoNogo_Stim, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'GoNogo_Stim.started')
                    # update status
                    GoNogo_Stim.status = STARTED
                    GoNogo_Stim.setAutoDraw(True)
                
                # if GoNogo_Stim is active this frame...
                if GoNogo_Stim.status == STARTED:
                    # update params
                    pass
                
                # if GoNogo_Stim is stopping this frame...
                if GoNogo_Stim.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > GoNogo_Stim.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        GoNogo_Stim.tStop = t  # not accounting for scr refresh
                        GoNogo_Stim.tStopRefresh = tThisFlipGlobal  # on global time
                        GoNogo_Stim.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'GoNogo_Stim.stopped')
                        # update status
                        GoNogo_Stim.status = FINISHED
                        GoNogo_Stim.setAutoDraw(False)
                
                # *Key_Resp* updates
                waitOnFlip = False
                
                # if Key_Resp is starting this frame...
                if Key_Resp.status == NOT_STARTED and tThisFlip >= CueTime+0.2-frameTolerance:
                    # keep track of start time/frame for later
                    Key_Resp.frameNStart = frameN  # exact frame index
                    Key_Resp.tStart = t  # local t and not account for scr refresh
                    Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Key_Resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Key_Resp.started')
                    # update status
                    Key_Resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(Key_Resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if Key_Resp is stopping this frame...
                if Key_Resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Key_Resp.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        Key_Resp.tStop = t  # not accounting for scr refresh
                        Key_Resp.tStopRefresh = tThisFlipGlobal  # on global time
                        Key_Resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Key_Resp.stopped')
                        # update status
                        Key_Resp.status = FINISHED
                        Key_Resp.status = FINISHED
                if Key_Resp.status == STARTED and not waitOnFlip:
                    theseKeys = Key_Resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _Key_Resp_allKeys.extend(theseKeys)
                    if len(_Key_Resp_allKeys):
                        Key_Resp.keys = _Key_Resp_allKeys[-1].name  # just the last key pressed
                        Key_Resp.rt = _Key_Resp_allKeys[-1].rt
                        Key_Resp.duration = _Key_Resp_allKeys[-1].duration
                        # was this correct?
                        if (Key_Resp.keys == str(CRESP)) or (Key_Resp.keys == CRESP):
                            Key_Resp.corr = 1
                        else:
                            Key_Resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Trial" ---
            for thisComponent in Trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Trial
            Trial.tStop = globalClock.getTime(format='float')
            Trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Trial.stopped', Trial.tStop)
            # check responses
            if Key_Resp.keys in ['', [], None]:  # No response was made
                Key_Resp.keys = None
                # was no response the correct answer?!
                if str(CRESP).lower() == 'none':
                   Key_Resp.corr = 1;  # correct non-response
                else:
                   Key_Resp.corr = 0;  # failed to respond (incorrectly)
            # store data for Trials (TrialHandler)
            Trials.addData('Key_Resp.keys',Key_Resp.keys)
            Trials.addData('Key_Resp.corr', Key_Resp.corr)
            if Key_Resp.keys != None:  # we had a response
                Trials.addData('Key_Resp.rt', Key_Resp.rt)
                Trials.addData('Key_Resp.duration', Key_Resp.duration)
            # the Routine "Trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Feedback" ---
            # create an object to store info about Routine Feedback
            Feedback = data.Routine(
                name='Feedback',
                components=[Feedback_Present],
            )
            Feedback.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from feedback_sentence
            if StimType == 'Go': # go cue 제시 조건에서 
                if Key_Resp.corr == 1: # 정답을 맞췄으면,
                    fb_color = 'Maroon'
                    fb_msg = "정확합니다"+"\n"+"반응속도: "+str(int(Key_Resp.rt*1000))+"ms"
                else: # 틀린 경우에
                    fb_color = 'MediumBlue'
                    fb_msg = "틀렸습니다"
            elif StimType == 'NoGo': # Nogo cue 제시 조건에서 
                if Key_Resp.corr == 1: # 정답을 맞췄으면,
                    fb_color = 'Maroon'
                    fb_msg = "정확합니다"
                else: # 틀린 경우에
                    fb_color = 'MediumBlue'
                    fb_msg = "틀렸습니다. 버튼을 누르면 안 됩니다."    
            Feedback_Present.setColor(fb_color, colorSpace='rgb')
            Feedback_Present.setText(fb_msg)
            # store start times for Feedback
            Feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Feedback.tStart = globalClock.getTime(format='float')
            Feedback.status = STARTED
            thisExp.addData('Feedback.started', Feedback.tStart)
            Feedback.maxDuration = None
            # keep track of which components have finished
            FeedbackComponents = Feedback.components
            for thisComponent in Feedback.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Feedback" ---
            # if trial has changed, end Routine now
            if isinstance(Trials, data.TrialHandler2) and thisTrial.thisN != Trials.thisTrial.thisN:
                continueRoutine = False
            Feedback.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Feedback_Present* updates
                
                # if Feedback_Present is starting this frame...
                if Feedback_Present.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    Feedback_Present.frameNStart = frameN  # exact frame index
                    Feedback_Present.tStart = t  # local t and not account for scr refresh
                    Feedback_Present.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Feedback_Present, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Feedback_Present.started')
                    # update status
                    Feedback_Present.status = STARTED
                    Feedback_Present.setAutoDraw(True)
                
                # if Feedback_Present is active this frame...
                if Feedback_Present.status == STARTED:
                    # update params
                    pass
                
                # if Feedback_Present is stopping this frame...
                if Feedback_Present.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Feedback_Present.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        Feedback_Present.tStop = t  # not accounting for scr refresh
                        Feedback_Present.tStopRefresh = tThisFlipGlobal  # on global time
                        Feedback_Present.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Feedback_Present.stopped')
                        # update status
                        Feedback_Present.status = FINISHED
                        Feedback_Present.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Feedback.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Feedback.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Feedback" ---
            for thisComponent in Feedback.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Feedback
            Feedback.tStop = globalClock.getTime(format='float')
            Feedback.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Feedback.stopped', Feedback.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Feedback.maxDurationReached:
                routineTimer.addTime(-Feedback.maxDuration)
            elif Feedback.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'Trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Blocks'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "TaskEnd" ---
    # create an object to store info about Routine TaskEnd
    TaskEnd = data.Routine(
        name='TaskEnd',
        components=[],
    )
    TaskEnd.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for TaskEnd
    TaskEnd.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    TaskEnd.tStart = globalClock.getTime(format='float')
    TaskEnd.status = STARTED
    thisExp.addData('TaskEnd.started', TaskEnd.tStart)
    TaskEnd.maxDuration = None
    # keep track of which components have finished
    TaskEndComponents = TaskEnd.components
    for thisComponent in TaskEnd.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "TaskEnd" ---
    TaskEnd.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            TaskEnd.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in TaskEnd.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "TaskEnd" ---
    for thisComponent in TaskEnd.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for TaskEnd
    TaskEnd.tStop = globalClock.getTime(format='float')
    TaskEnd.tStopRefresh = tThisFlipGlobal
    thisExp.addData('TaskEnd.stopped', TaskEnd.tStop)
    thisExp.nextEntry()
    # the Routine "TaskEnd" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
