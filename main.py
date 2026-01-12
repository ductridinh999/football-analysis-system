from utils import read_video, save_video
from trackers import Tracker
import cv2
import copy
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import copy

def main():
    # Read Video
    video_frames = read_video('input/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    # Pass Tracking Variables
    current_passes = {
        1: {'total': 0, 'success': 0},
        2: {'total': 0, 'success': 0}
    }
    pass_stats_per_frame = [] 

    current_possessor_candidate = -1
    possession_frame_count = 0
    POSSESSION_THRESHOLD = 5  # Frames a player must hold ball to confirm possession

    last_confirmed_possessor = -1
    last_confirmed_team = -1

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Filter Noise (Wait for Threshold)
        if assigned_player != -1:
            if assigned_player == current_possessor_candidate:
                possession_frame_count += 1
            else:
                # New candidate, reset counter
                current_possessor_candidate = assigned_player
                possession_frame_count = 1
        else:
            # Ball is loose/in-air. Reset counter.
            current_possessor_candidate = -1
            possession_frame_count = 0

        # Confirm Possession Change
        if possession_frame_count >= POSSESSION_THRESHOLD:
            # We have a valid possession!
            confirmed_player = current_possessor_candidate
            confirmed_team = tracks['players'][frame_num][confirmed_player]['team']

            if confirmed_player != last_confirmed_possessor:
                
                # Count Pass (only if we had a previous holder)
                if last_confirmed_possessor != -1:
                    passing_team = last_confirmed_team
                    
                    # Increment Total Pass
                    current_passes[passing_team]['total'] += 1
                    
                    # Increment Success if teams match
                    if confirmed_team == passing_team:
                        current_passes[passing_team]['success'] += 1
                
                # Update the "Last Confirmed" state
                last_confirmed_possessor = confirmed_player
                last_confirmed_team = confirmed_team

        # Visual
        if possession_frame_count >= POSSESSION_THRESHOLD:
             tracks['players'][frame_num][current_possessor_candidate]['has_ball'] = True
        
        # Team Control Logic:
        # If ball is loose, control stays with the last team that had it (Possession Rule)
        if last_confirmed_team != -1:
            team_ball_control.append(last_confirmed_team)
        else:
            team_ball_control.append(0) # No one has established control yet

        pass_stats_per_frame.append(copy.deepcopy(current_passes))

    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control, pass_stats_per_frame)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, 'output/output_video.mp4')

if __name__ == '__main__':
    main()