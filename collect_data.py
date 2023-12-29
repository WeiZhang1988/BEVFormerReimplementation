#!/usr/bin/env python

import glob
import os
import sys
import time

try:
    sys.path.append('/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
except IndexError:
    pass

import carla
import random
import numpy as np
import cv2
from queue import Queue, Empty
import copy
import random
random.seed(0)

# args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--host', metavar='H',    default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p',           default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port',              default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--ego-spawn', type=list, default=None, help='[x,y] in world coordinate')
parser.add_argument('--top-view',             default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map',                  default='Town10HD', help='Town Map')
parser.add_argument('--sync',                 default=True, help='Synchronous mode execution')
parser.add_argument('--sensor-h',             default=2.4, help='Sensor Height')
# absolute path!!!
parser.add_argument('--save-path',            default='/home/ubuntu/Downloads/Carla_dataset/', help='Synchronous mode execution')
parser.add_argument('--number-of-vehicles', metavar='N',   default=30, help='Number of vehicles (default: 60)')
parser.add_argument('--number-of-walkers', metavar='W',    default=10, help='Number of walkers (default: 30)')
parser.add_argument('--res',    metavar='WIDTHxHEIGHT',    default='1280x720', help='window resolution (default: 1280x720)')

args = parser.parse_args()
args.width, args.height = [int(x) for x in args.res.split('x')]

# picture size
IM_WIDTH = 256
IM_HEIGHT = 256

actor_list, sensor_list = [], []
vehicles_list, walkers_list = [], []
walkers_id = []
sensor_type = ['rgb','ins']
def main(args):
    # start creating the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    
    #world = client.get_world()
    world = client.load_world('Town10HD')
    blueprint_library = world.get_blueprint_library()
    try:
        original_settings = world.get_settings()
        settings = world.get_settings()

        # traffic manager
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)
        # ignore traffic lights
        # tm.ignore_lights_percentage(ego_vehicle, 100)
        # if speed limit 30km/h -> 30*(1-10%)=27km/h
        tm.global_percentage_speed_difference(10.0)

        # setting CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        # settings.no_rendering_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        # manual setting
        # transform_vehicle = carla.Transform(carla.Location(0, 10, 0), carla.Rotation(0, 0, 0))
        # auto select
        ego_blueprints = world.get_blueprint_library().filter("model3")
        vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
        walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            args.number_of_vehicles = number_of_spawn_points

        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(vehicle_blueprints)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.spawn_actor(blueprint, transform)
            vehicle.set_autopilot(True, tm.get_port())
            vehicles_list.append(vehicle)
        transform_vehicle = random.choice(spawn_points)
        ego_blueprint = random.choice(ego_blueprints)
        ego_blueprint.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_blueprint, transform_vehicle)
        ego_vehicle.set_autopilot(True, tm.get_port())
        actor_list.append(ego_vehicle)

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        SpawnActor = carla.command.SpawnActor
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            walkers_list.append({"id": results[i].actor_id})
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            walkers_id.append(walkers_list[i]["con"])
            walkers_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(walkers_id)
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()
        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(0.0)
        for i in range(0, len(walkers_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())

        #--------------------------Sensor settings--------------------------#
        sensor_queue = Queue()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        # lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        semantic_bp = blueprint_library.find('sensor.camera.instance_segmentation')

        # set the attribute of camera
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute('sensor_tick', '0.1')

        semantic_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        semantic_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        semantic_bp.set_attribute("fov", "120")
        semantic_bp.set_attribute('sensor_tick', '0.1')

        '''# cam top only
        cam_top_bp = blueprint_library.find('sensor.camera.rgb')
        cam_top_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_top_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_top_bp.set_attribute("fov", "120")
        
        cam00 = world.spawn_actor(cam_top_bp, carla.Transform(carla.Location(3.0,0,10.0),carla.Rotation(yaw=0,pitch=-90,roll=0)), attach_to=ego_vehicle)
        cam00.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_top"))
        sensor_list.append(cam00)
        '''
        
        cam01 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h),carla.Rotation(yaw=0)), attach_to=ego_vehicle)
        cam01.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_front"))
        sensor_list.append(cam01)

        cam02 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h),carla.Rotation(yaw=90)), attach_to=ego_vehicle)
        cam02.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_right"))
        sensor_list.append(cam02)
        
        cam03 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h),carla.Rotation(yaw=180)), attach_to=ego_vehicle)
        cam03.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_behand"))
        sensor_list.append(cam03)

        cam04 = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=args.sensor_h),carla.Rotation(yaw=270)), attach_to=ego_vehicle)
        cam04.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_left"))
        sensor_list.append(cam04)

        semantic_cam = world.spawn_actor(semantic_bp, carla.Transform(carla.Location(5.0,0,10.0),carla.Rotation(yaw=0,pitch=-90,roll=0)), attach_to=ego_vehicle)
        semantic_cam.listen(lambda data: sensor_callback(data, sensor_queue, "ins_cam"))
        sensor_list.append(semantic_cam)

        '''lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '200000')
        lidar_bp.set_attribute('range', '32')
        lidar_bp.set_attribute('rotation_frequency', str(int(1/settings.fixed_delta_seconds)))
        
        lidar01 = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=args.sensor_h)), attach_to=ego_vehicle)
        lidar01.listen(lambda data: sensor_callback(data, sensor_queue, "lidar"))
        sensor_list.append(lidar01)
        '''

		#--------------------------Data collection--------------------------#
        total_tick = 6000
        while total_tick > 0:
            total_tick -= 1
            # tick the server
            world.tick()

            # follow the CARLA interface camera as the car moves
            loc = ego_vehicle.get_transform().location
            spectator.set_transform(carla.Transform(carla.Location(x=loc.x,y=loc.y,z=35),carla.Rotation(yaw=0,pitch=-90,roll=0)))

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            try:
                rgbs = []
                sem_ins = []
                # sem_rgb = []

                for i in range (0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        rgbs.append(_parse_image_cb(s_data))
                        if s_name.split('_')[1] == 'front':
                            rgb_0 = _parse_image_cb(s_data)[...,:3]
                        elif s_name.split('_')[1] == 'left':
                            rgb_1 = _parse_image_cb(s_data)[...,:3]
                        elif s_name.split('_')[1] == 'behand':
                            rgb_2 = _parse_image_cb(s_data)[...,:3]
                        elif s_name.split('_')[1] == 'right':
                            rgb_3 = _parse_image_cb(s_data)[...,:3]
                        # if s_name.split('_')[1] == 'top':
                            # sem_rgb.append(_parse_image_cb(s_data))
                    elif sensor_type == 'ins':
                        rgbs.append(_parse_instance_semantic_cb(s_data))
                        ins_sem = _parse_instance_semantic_cb(s_data)[...,:3]
                    #elif sensor_type == 'sem':
                    #    rgbs.append(_parse_semantic_cb(s_data))
                    #elif sensor_type == 'lidar':
                    #    lidar = _parse_lidar_cb(s_data)
                
                # visualization
                vis_rgb=np.concatenate(rgbs, axis=1)[...,:3]
                # ins_sem=sem_ins[0][...,:3]
                # rgb_sem=sem_rgb[0][...,:3]
                cv2.imshow('vizs', visualize_data(vis_rgb, None))
                cv2.waitKey(100)

                mkdir_folder(args.save_path)
                if vis_rgb is None or args.save_path is not None:
                    filename0 = args.save_path +'images/'+format(w_frame,'05d')+'_'+'00'+'.png'
                    cv2.imwrite(filename0, np.array(rgb_0))
                    filename1 = args.save_path +'images/'+format(w_frame,'05d')+'_'+'01'+'.png'
                    cv2.imwrite(filename1, np.array(rgb_1))
                    filename2 = args.save_path +'images/'+format(w_frame,'05d')+'_'+'02'+'.png'
                    cv2.imwrite(filename2, np.array(rgb_2))
                    filename3 = args.save_path +'images/'+format(w_frame,'05d')+'_'+'03'+'.png'
                    cv2.imwrite(filename3, np.array(rgb_3))
                    # filename = args.save_path +'lidar/'+format(w_frame,'05d')+'.npy'
                    # np.save(filename, lidar)
                if ins_sem is None or args.save_path is not None:
                    filename = args.save_path +'instance_semantics/'+format(w_frame,'05d')+'.png'
                    cv2.imwrite(filename, np.array(ins_sem[...,::-1]))
                #if rgb_sem is None or args.save_path is not None:
                #    filename = args.save_path +'ins/'+'rgb'+format(w_frame,'05d')+'.png'
                #    cv2.imwrite(filename, np.array(rgb_sem[...,::-1]))

            except Empty:
                print("Some of the sensor information is missed")

    finally:
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)
        for sensor in sensor_list:
            sensor.destroy()
        for actor in actor_list:
            actor.destroy()
        for vehicle in vehicles_list:
            vehicle.destroy()
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(walkers_id), 2):
            all_actors[i].stop()

        client.apply_batch([carla.command.DestroyActor(x) for x in walkers_id])
        time.sleep(0.1)
        print("All cleaned up!")

def mkdir_folder(path):
    for s_type in sensor_type:
        if s_type == 'rgb':
            if not os.path.isdir(os.path.join(path, 'images')):
                os.makedirs(os.path.join(path, 'images'))
        if s_type == 'ins':
            if not os.path.isdir(os.path.join(path, 'instance_semantics')):
                os.makedirs(os.path.join(path, 'instance_semantics'))
    return True

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))

# modify from world on rail code
def visualize_data(rgb, lidar, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):

    # canvas = np.array(rgb[...,::-1])
    canvas = np.array(rgb)

    if lidar is not None:
        lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz,cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([canvas, cv2.resize(lidar_viz.astype(np.uint8), (canvas.shape[0], canvas.shape[0]))], axis=1)

    # cv2.putText(canvas, f'yaw angle: {imu_yaw:.3f}', (4, 10), *text_args)
    # cv2.putText(canvas, f'log: {gnss[0]:.3f} alt: {gnss[1]:.3f} brake: {gnss[2]:.3f}', (4, 20), *text_args)

    return canvas
# modify from world on rail code
def lidar_to_bev(lidar, min_x=-24,max_x=24,min_y=-16,max_y=16, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care rgbabout.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1,:]

# modify from manual control
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # array = array[:, :, ::-1]
    return array
# modify from leaderboard
def _parse_lidar_cb(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points
# modify from manual control
def _parse_semantic_cb(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    # BGR2RGB [:,:,[2,1,0]]
    array = array[:, :, :3]
    return array
def _parse_instance_semantic_cb(image):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array

if __name__ == "__main__":
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
