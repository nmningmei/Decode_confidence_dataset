Contributors:
Polina Arbuzova (polina.arbuzova@bccn-berlin.de), Caroline Peters (caroline.peters@bccn-berlin.de), Lukas Röd (lukas.roed@posteo.net), Christina Koß (christina.koss@bccn-berlin.de), Elisa Filevich (elisa.filevich@gmail.com)

Citation:
in prep.

Stimulus:
Two trajectories are shown, one of which is the real trajectory (the one participants produced with their throw) and the other one an altered one (altered speed at release resulting in different trajectory). The task is to determine the real trajectory (2AFC task). There are two possible stimuli (i.e. positions of the real trajectory): 1 = right, 2 = left. [*See section "Skittles task" at the end of this readme file.*]

Difference:
This relates to trial difficulty. It is the difference in release velocities for the real and the altered trajectory, also referred to as vdiff in this readme. Unit: m/s. 

Confidence scale:
A vertical visual analogue scale. Confidence ratings (referring to the type 1 response) go continuously from "unsicher" (unsure, coded as 0) to "sicher" (sure, coded as 100).

Manipulations:
There were three conditions. 1 = visuomotor (active throw, the ball was shown during throw), 2 = motor (active throw, the ball was not shown during throw), 3 = visual (no active throw, a pseudorandom throw from the previous active block is shown as a video).

Block size:
3 blocks of 108 trials each (36 per condition).

Feedback:
No feedback was given.

NaN fields:
Subjects had the option to mark trials as "error", e.g. if they accidentally selected a different trajectory than intended. These trials are excluded.
Visual and visuomotor trials where one trajectory hit the central post and the other one didn't were excluded, because these were artificially easier than the difference in release velocity would suggest.
Trials where one trajectory went to the right of the central post and the other one to the left were excluded, because they were artificially easier than the difference in release velocity would suggest.
For excluded trials, the values of all response variables (Response, Confidence, RT_dec, RT_conf) are set to NaN.

Subject population:
mean age of 26.9 years, standard deviation 4.1 years

Response device:
A lever that was also used to perform the motor task.

Experiment setting:
In lab.

Training:
50 training trials (visuomotor and motor condition only, type 1 task only, no confidence rating).
8 feedback trials (visuomotor and motor condition only, with confidence rating and feedback about correctness of type 1 response after giving the confidence rating).

Experiment goal:
Investigate correlations between metacognition in visual, visuomotor and motor condition.

Special instructions:
Subjects were instructed to use the full confidence scale.

Experiment dates:
February - June 2018

Location of data collection:
Metamotor Lab, Humboldt-University Berlin, Germany

Any other important information about the dataset:

Skittles task: 
In the Skittles task, subjects sit at a table next to a lever: a horizontal metal bar that rotates around its axis. The angle of the lever is recorded by a potentiometer. At the far end of the lever, a contact sensor detects whether the subject is resting their index finger on the tip of the lever (as if holding a ball) or has released it (as if having thrown a ball). Data are read from the lever into a computer using a Labjack T7 (Labjack, Lakewood, CO) at a sampling rate of 1 kHz. Subjects rest their elbow on the rotation axis of the bar so that they can rotate the lever. On a screen in front of them, subjects see a short bar that moves in one-to-one correspondence to the lever (from the birds-eye viewpoint). When the contact sensor at the end of the lever is touched, a ball is displayed at the tip of the bar on the screen. By moving the bar and lifting their index finger from the contact sensor, subjects "throw" the ball around a pole, attempting to hit a target (Skittle) standing behind it. As soon as subjects release the ball, the target disappears such that they do not see if they hit the target or not. After each ball throw, subjects see two possible ball trajectories displayed on the screen and have to choose which of them they produced with their throw.
Depending on the condition, different information was available during the task and at type 1 response (see the description of the conditions above). 

Difference in release velocity:
Participants threw the virtual ball at a specific speed. In order to achieve a 2AFC task, a second, alternative trajectory was produced and participants chose between the two. The alternative trajectory was produced by altering the speed at release of the ball. A larger speed difference between real and alternative trajectory leads to easier trials since the resulting trajectories are further apart from each other (and vice versa for smaller speed differences). Hence, bigger vdiff values correspond to easier discrimination trials. 
We could therefore control task difficulty by staircasing the speed difference between real and alternative trajectory in a 2 up - 1 down manner to reach approximately 71% correct.
