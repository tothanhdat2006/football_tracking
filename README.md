# Football tracking
A project about using Deep Learning model to analyze track and analyze football matches
## What?
This project focuses on building a tracking system used in football analysis. It is capabale of:

- [ ] Detecting players, goalkeepers, referees and balls (in progress)
- [ ] Keeping track of player ID (in progress)
- [ ] Map of players
- [ ] Heatmap of players 
- [ ] Tracking ball movement (in progress)
- [ ] Analyze players' ball time (in progress)

## Idea
- Using yolo11 to detect players, goalkeepers, referees and balls
- Using supervision for custom annotator
- Using yolo11-pose to detect field and create homography matrix -> map camera perspective to top-down perspective
- Using bytetrack to track ball movement
- Using seaborn to create heatmap of players

## How to use

To track a video:

`python track.py --source="SOURCE_OF_VIDEO" --dest="RESULT_DESTINATION"`

## Reference

- [Roboflow sports repo](https://github.com/roboflow/sports)
- [Yolo11](https://docs.ultralytics.com/models/yolo11/)
