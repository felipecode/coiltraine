


def plot_episode(city_name, file_name, color_palate):


    # This can be hardcoded since the map file name is always the same
    image_location = map.__file__[:-7]
    print ("Image logatio n", image_location )

    map_image = Image.open(os.path.join(image_location, city_name + '.png'  ))
    map_image.load()
    map_image = np.asarray(map_image, dtype="int32")


    carla_map = map.CarlaMap(city_name, 0.164, 50)

    f = open(file_name, "rU")
    header_details = f.readline()

    header_details = header_details.split(',')
    header_details[-1] = header_details[-1][:-2]
    f.close()

    details_matrix = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)


    #
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]

    count_episodes = 0
    episode_palete = [0]
    travelled_distance = []
    travel_this_episode = 0
    print ("len details  ", len(details_matrix))
    for i in range(len(details_matrix)):
        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]
        print (sldist(point, previous_pos))

        if sldist(point, previous_pos) > 100.0:
            count_episodes += 1
            travelled_distance.append(travel_this_episode)
            travel_this_episode = 0
            episode_palete.append(i)
            previous_pos = point
        if count_episodes == number_of_episodes:
            break

        travel_this_episode += sldist(point, previous_pos)
        previous_pos = point


    print (episode_palete)

    count_episodes = 1
    previous_pos = [details_matrix[0, header_details.index('pos_x')],
                 details_matrix[0, header_details.index('pos_y')]]

    for i in range(0, episode_palete[-1]):

        point = [details_matrix[i, header_details.index('pos_x')],
                 details_matrix[i, header_details.index('pos_y')]]


        if sldist(point,previous_pos) > 500.0: # DUMB BUT WHATEVER
            count_episodes += 1
            travel_this_episode = 0
            previous_pos = point
        if count_episodes == number_of_episodes+1:
            break

        travel_this_episode += sldist(point, previous_pos)

        previous_pos = point


        value = travel_this_episode/travelled_distance[count_episodes-1]

        color_palate_inst = [0+(value*x) for x in color_palate[count_episodes-1][:-1]]
        color_palate_inst.append(255)

        point.append(0.0)

        print ("palete chosen ", color_palate_inst)
        plot_on_map(map_image, carla_map.convert_to_pixel(point), color_palate_inst, 4)


    plot_test_image(map_image, 'test.png')
