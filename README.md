# Football tracking
A project about using Deep Learning model to analyze track and analyze football matches
## What?
This project focuses on building a tracking system used in football analysis. What I plan to achive:

- [x] Detecting players, goalkeepers, referees and balls               (easy)
- [x] Keeping track of players' ID                                     (easy)
- [x] Tracking ball movement                                           (easy)
- [ ] Analyzing players' ball time                                     (medium)
- [ ] Re-identification after transition                               (hard)
- [ ] Predict players' movement on map                                 (hard) 
- [x] Map of players                                                   (hard)
- [ ] Heatmap of players                                               (hard)

## Ideas
- Using yolo11 to detect players, goalkeepers, referees and balls
- Using supervision for custom annotator
- Using yolo11-pose to detect field and create homography matrix -> map camera perspective to top-down perspective
- Using bytetrack to track ball movement
- Using seaborn to create heatmap of players

## Challenges
- Different camera angles
- Abrupt changes in camera view
- Players exchange (re-identification problem)
- Players' numbers are not visible
- Unrelated scene (logo transition, audience, slow-mo replays,...)

## How to use

To track a video:

`python track.py --source="SOURCE_OF_VIDEO" --dest="RESULT_DESTINATION"`

## Reference

- [Roboflow sports repo](https://github.com/roboflow/sports)
- [Yolo11](https://docs.ultralytics.com/models/yolo11/)
