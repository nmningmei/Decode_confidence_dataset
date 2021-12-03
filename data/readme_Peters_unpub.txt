Contributors:
Caroline Peters (Caroline.peters@bccn-berlin.de), Elisa Filevich (elisa.filevich@gmail.com)

Citation:
in prep.

Stimulus:
Two trajectories are shown, one of which is the real trajectory (the one participants produced with their throw) and the other one an altered one (altered speed at release resulting in different trajectory). The task is to determine the real trajectory (2AFC task). There are two possible stimuli (i.e. positions of the real trajectory): 1 = right, 2 = left. [*See section "Skittles task" at the end of this readme file.*]

Difference:
This relates to trial difficulty. It is the difference in release velocities for the real and the altered trajectory, also referred to as vdiff in this readme. Unit: m/s. 

Confidence scale:
A vertical visual analog scale. Confidence ratings (referring to the type 1 response) go continuously from "unsicher" (unsure, coded as 0) to "sicher" (sure, coded as 100).

Manipulations:
There were two conditions. 1 = same outcome (either both trajectories hit the target or neither trajectory hits the target), 2 = different outcome (one of the trajectories hits the target, the other does not).
The conditions were realised by altering the speed at release of the alternative trajectory in such a way that the condition needs are met. 

Block size:
4 blocks with 120 trials each (60 per condition, 480 trials in total).

Feedback:
No feedback was given.

NaN fields:
Subjects had the option to mark trials as "error", e.g. if they accidentally selected a different trajectory than intended. These trials are excluded.
Trials where one trajectory hits the central post and the other one doesn't were excluded, because these were artificially easier than the difference in release velocity would suggest (for an explanation of difference in release velocity see "other important information").
Trials where one trajectory goes to the right of the central post and the other one to the left were excluded, because they were artificially easier than the difference in release velocity would suggest.
For excluded trials, the values of all response variables (Response, Confidence, RT_dec, RT_conf) are set to NaN.

Subject population:
mean age 26.3 years with a standard deviation of 4.4 years

Response device:
A lever that was also used to perform the motor task.

Experiment setting:
In lab.

Training:
56 training trials (type 1 task only, no confidence rating).
16 feedback trials (with confidence rating and feedback about correctness of type 1 response after giving the confidence rating).

Experiment goal:
Confirm whether metacognitive efficiency as measured by M-ratio (meta-d'/d') is higher for the "different outcome" condition than for the "same outcome" condition.
This would provide evidence for the hypothesis that monitoring the outcome of a movement is informative for the metacognitive judgement (in the "different outcome" condition, the outcome provides additional information that can potentially be used for the metacognitive judgement). 

Special instructions:
Subjects were instructed to use the full confidence scale and to rate trials relative to each other.

Experiment dates:
January - March 2019

Location of data collection:
Metamotor Lab, Humboldt-University Berlin, Germany


Any other important information about the dataset:

Short description of the "Skittles task":
In the Skittles task, subjects sit next to a manipulandum or lever: a horizontal metal bar that rotates around its axis. The angle of the lever is recorded by means of a potentiometer. At the far end of the lever, a contact sensor detects whether the subject is resting their index finger on the tip of the lever (as if holding a ball) or has released it (as if having thrown a ball). Data are read from the lever into a computer using a Labjack T7 (Labjack, Lakewood, CO) at a sampling rate of 1 kHz. Subjects rest their elbow on the rotation axis of the bar so that extending their arm rotates the bar to the right. On a screen in front of them, subjects see a short bar that moves corresponding to the lever. When the contact sensor at the end of the lever is touched, a ball is displayed at the tip of the bar on the screen. By moving the bar and lifting their index finger from the contact sensor, subjects “throw” the ball around a pole, attempting to hit a target (Skittle) standing behind it. As soon as subjects let go of the ball, the target disappears such that they don’t see if the outcome of the throw is a hit or not.
After each ball throw, subjects see two possible ball trajectories displayed on the screen. Depending on the condition, they will either have the same endpoints (same condition, either both hitting the target or both not hitting the target) or different endpoints (different condition, one hits the target, the other one does not). 

Difference in release velocity:
Participants threw the virtual ball at a specific speed. In order to achieve a 2AFC task, a second, alternative trajectory was produced and participants chose between the two. The alternative trajectory was produced by altering the speed at release of the ball. A larger speed difference between real and alternative trajectory leads to easier trials since the resulting trajectories are further apart from each other (and vice versa for smaller speed differences). Hence, the bigger vdiff gets the easier it gets for participants to choose the trajectory they actually threw themselves. 
We could therefore control task difficulty by staircasing the speed difference between real and alternative trajectory in a 2 up - 1 down manner to reach 71% correct.

Staircase procedure:
In order to achieve a balanced hit/no hit behavior of subjects’ throws, the ball and target sizes are adapted with a 1 up – 1 down staircase which stabilizes performance at approximately 50% hits.
The velocity difference vdiff is adapted with a 2 up - 1 down staircase in order to achieve a performance of approximately 71% correct in the type-1 task. However, if using the staircased vdiff doesn’t allow for producing the required condition in that trial ("same outcome" or "different outcome"), the nearest vdiff that does is chosen. 
For our purposes, we excluded trials where the vdiff chosen in such a way was off more than 2 standard deviations from the mean of the ten last staircased vdiffs since task difficulty was easier than it should have been according to the staircase.
In the present data set, however, we did not exclude those trials: Whoever wants to work with the data, can decide what to exclude.