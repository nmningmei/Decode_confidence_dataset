Contributors:
Polina Arbuzova (polina.arbuzova@bccn-berlin.de), Elisa Filevich (elisa.filevich@gmail.com)

Citation:
in prep.

Tasks:
Subjects were tested on two different tasks that were done in two separate sessions on different days. The order of tasks was pseudo-randomized between subjects. 21 subjects completed task 1 ('Trajectories') first, 20 subjects completed task 2 ('Angles') first. 
Two subjects (#25 & #33) only completed task 1 ('Trajectories') and were excluded from the second task due to the performance beyond the desired range (60-80% correct) in two or more conditions.
Task 1 ('Trajectories'): Subjects should decide which trajectory belongs to the preceding throw.
Task 2 ('Angles'): Subjects should decide which angle of release belongs to the preceding throw.
In both tasks, the difficulty was staircased with a 2-up-1-down staircase procedure to ensure type-1 performance around 71%. [*See section "Skittles task" at the end of this readme file.*]

Stimulus:
For task 1 ('Trajectories'): Two trajectories are shown, one of which is the real trajectory and the other one an altered one. The task is to determine the real trajectory. There are two possible stimuli (i.e. relative positions of the real trajectory on the screen): 1 = right, 2 = left.
For task 2 ('Angles'): Two levers are shown, one of which has the real angle at the moment of the ball release and the other one an altered one. The task is to determine lever with the real angle. There are two possible stimuli (i.e. relative positions of the real lever on the screen): 1 = right, 2 = left.

Difference:
This relates to trial difficulty. Values can only be compared within task, as they have different meanings for task 1 & task 2.
For task 1 ('Trajectories'), this is the difference in release velocities for the real and the altered trajectory. Unit: m/s. 
For task 2 ('Angles'), this is the difference in angle between the real and the altered lever position. Unit: degrees (°). 

Confidence scale:
Discrete confidence scale with ratings from 1 (lowest) to 6 (highest), with a step of 1. 

Manipulations:
There were three conditions. 1 = visuomotor (active throw, ball and lever were shown during the throw), 2 = motor (active throw, the critical part of the display was not shown during the throw: ball (in trajectories) or lever (in angles)), 3 = visual (no active throw, a pseudorandom throw from the previous active block is shown as a video). 

Block size:
For each task, 5 blocks with 120 trials each (40 per condition).
One subject (#27) only completed 4 blocks for task 2 (angles) due to technical issues and time constraints.

Feedback:
After each block, subjects were told how many times they hit the target. No feedback on the type-1 response was given.

NaN fields:
Subjects had 10s time to respond for the type-1 question. When they failed to respond within that time frame, the trial was excluded.
Subjects had the option to mark trials as "error", e.g. if they accidentally selected a different trajectory than intended. These trials are excluded.
For task 1 ('Trajectories'), trials where one trajectory went to the right of the central post and the other one to the left were excluded, because they were artificially easier than the difference in release velocity would suggest.
For task 1 ('Trajectories') in the visuomotor and visual condition, trials where one of the displayed trajectories hit the central post while the other one didn't were excluded, because they were artificially easier than the difference in release velocity would suggest.
Such trials also were not taken into account for online staircasing.
For excluded trials, the values of all response variables (Response, Confidence, RT_dec, RT_conf) are set to NaN.

Subject population:
Values are given for participants who completed both tasks.
Range of ages: 20 to 34 years. Mean age: 27.17 years (SD: 4.07 years); 31 female participants.

Response device:
Keyboard.

Experiment setting:
In lab.

Training:
8 "pre-training" trials (visuomotor, only throwing, no type-1 or type-2 question)
8 feedback trials (visuomotor & motor, type-1 task only, with feedback on type-1 response)
8 confidence & feedback trials (visuomotor & motor, type-1 & type-2 question, with feedback on type-1 task)
96 "training" trials (visuomotor & motor, type-1 task only, without feedback)

Experiment goal:
Investigate correlations between metacognitive efficiency (measured as M-ratio) in the different conditions and tasks:
- Confirm if the correlation pattern between the three conditions is the same for the angles task as for the trajectories task. (In a previous experiment with only the trajectories task, there was a correlation between visual and visuomotor trials but not between visual and motor or visuomotor and motor);
- Confirm if there are correlations between the respective conditions of the two tasks (visuomotor-angles & visuomotor-trajectories, motor-angles & motor-trajectories, and visual-angles & visual-trajectories).
- Compare metacognitive efficiency between the motor conditions of the two tasks, to explore the possibility that subjects have better metacognitive access to proximal parameters of a movement (e.g. the angle of their arm in task 2) than to distal ones (e.g. the trajectory in task 1).

Special instructions:
Subjects were instructed to use the full confidence scale and be as precise in their confidence ratings as possible.
Subjects were not told explicitly about the 10 s runout time for type-1 response, although, they were encouraged not too spend too much time on each trial and were reminded about the total number of trials (600) in the main part. 

Experiment dates: 15.01.2019-16.04.2019

Location of data collection:
Metamotor lab, Humboldt-Universität zu Berlin, Germany

Any other important information about the dataset:
Skittles task: 
In the Skittles task, subjects sit at a table next to a lever: a horizontal metal bar that rotates around its axis. The angle of the lever is recorded by a potentiometer. At the far end of the lever, a contact sensor detects whether the subject is resting their index finger on the tip of the lever (as if holding a ball) or has released it (as if having thrown a ball). Data are read from the lever into a computer using a Labjack T7 (Labjack, Lakewood, CO) at a sampling rate of 1 kHz. Subjects rest their elbow on the rotation axis of the bar so that they can rotate the lever. On a screen in front of them, subjects see a short bar that moves in one-to-one correspondence to the lever (from a birds-eye viewpoint). When the contact sensor at the end of the lever is touched, a ball is displayed at the tip of the bar on the screen. By moving the bar and lifting their index finger from the contact sensor, subjects “throw” the ball around a pole, attempting to hit a target (Skittle) standing behind it. As soon as subjects release the ball, the target disappears such that they don’t see if they hit the target or not.
Depending on the condition and task, different information was available during the task and at type 1 response (see the description of the tasks and conditions above). 

Difference in release velocity and angle:
Ball throw was determined by two parameters: angle and the velocity at the time of the ball release. To obtain an alternative trajectory in task 1, we manipulated the velocity only.
In task 2, to obtain an alternative angle, we manipulated the angle directly.