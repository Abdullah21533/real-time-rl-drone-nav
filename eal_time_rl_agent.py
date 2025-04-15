# Step 1: Install Required Libraries
!pip install gym stable-baselines3 matplotlib imageio moviepy pillow

# Step 2: Import Libraries
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from PIL import Image  # Add this import for image resizing

# Step 3: Create and Train the Environment
env = make_vec_env('CartPole-v1', n_envs=1)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10_000)

# Step 4: Generate Enhanced Animation with Graph Overlay
def generate_enhanced_animation(env, model, filename="enhanced_animation.mp4", num_steps=500):
    frames = []
    obs = env.reset()
    
    # Create a figure for the graph overlay
    fig, ax = plt.subplots(figsize=(6, 4))
    rewards_over_time = []
    x_data = []
    
    for step in range(num_steps):
        # Render the environment frame
        frame = env.render(mode='rgb_array')
        
        # Predict action and step through the environment
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # Update rewards for the graph
        rewards_over_time.append(reward)
        x_data.append(step)
        
        # Plot the rewards graph
        ax.clear()
        ax.plot(x_data, rewards_over_time, color='blue', label="Reward")
        ax.set_title("Real-Time Rewards")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)
        
        # Save the graph as an image
        fig.canvas.draw()  # Draw the figure
        graph_image = np.array(fig.canvas.renderer.buffer_rgba())  # Convert to numpy array
        
        # Resize the graph image to match the environment frame height
        graph_height = frame.shape[0]
        graph_width = int(graph_height * (graph_image.shape[1] / graph_image.shape[0]))
        graph_image_resized = np.array(Image.fromarray(graph_image).resize((graph_width, graph_height)))
        
        # Combine the environment frame and graph
        combined_frame = np.hstack((frame, graph_image_resized[:, :, :3]))  # Remove alpha channel
        frames.append(combined_frame)
        
        # Reset if the episode ends
        if done:
            obs = env.reset()
    
    # Save frames as a video
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(filename, codec="libx264")
    print(f"Enhanced animation saved as '{filename}'")

# Generate the enhanced animation
generate_enhanced_animation(env, model)

# Step 5: Add Background Music to the Video
def add_background_music(video_file, audio_file, output_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    
    # Adjust audio duration to match video
    audio_clip = audio_clip.subclip(0, min(video_clip.duration, audio_clip.duration))
    
    # Add audio to video
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_file, codec="libx264")
    print(f"Video with music saved as '{output_file}'")

# Download a royalty-free background music file (or upload your own)
!wget -O background_music.mp3 https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3

# Add background music to the enhanced animation
add_background_music("enhanced_animation.mp4", "background_music.mp3", "final_animation_with_music.mp4")
