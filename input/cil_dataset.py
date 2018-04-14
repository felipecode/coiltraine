
from torch.utils.data import Dataset, DataLoader



# TODO: Warning, maybe this does not need to be included everywhere.
from configs import g_conf

class CILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, transform=None): # The transformation object.
        """
        Function to encapsulate the dataset

        Arguments:
            dataset_configuration: the configuration file for the datasets.
            root_dir (string): Directory with all the hdfiles from the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images = pre_load_hdf5_files(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, used_ids):


        # We test here directly and include the other images here.

        for s in range(len(self.images)):
            count = 0

            for chosen_key in used_ids:

                count_seq = 0
                first_enter = True

                for i in range(self._number_frames_fused):
                    chosen_key = chosen_key + i * 3

                    for es, ee, x in images[s]:

                        if chosen_key >= es and chosen_key < ee:
                            """ We found the part of the data to open """
                            # print x[]
                            first_enter = False

                            pos_inside = chosen_key - es

                            # print 'el i'
                            # print chosen_key
                            # print pos_inside
                            # print x[chosen_key - es - 1 + 1:chosen_key - es + 1,:,:,:].shape

                            sensors_batch[s][count, :, :,
                            (i * 3):((i + 1) * 3)] = np.array(x[pos_inside, :, :, :])

                            # print sensors_batch[s][count].shape
                            # if not self._perform_sequential:
                            # img = Image.fromarray(sensors_batch[s][count])
                            # img.save('test' + str(self._current_position_on_dataset +count) + '_0_.jpg')
                count += 1



    def pre_load_hdf5_files(self):

        datasets_cat = [list([]) for _ in xrange(len(dataset_names))]

        images_data_cat = [list([]) for _ in xrange(len(image_dataset_names))]

        lastidx = 0
        count = 0
        # print file_names
        for cword in file_names:
            try:
                print
                cword
                print
                count
                dset = h5py.File(cword, "r")

                for i in range(len(image_dataset_names)):
                    # print image_dataset_names[i]
                    x = dset[image_dataset_names[i]]
                    # print x
                    old_shape = x.shape[0]
                    # print old_shape

                    images_data_cat[i].append((lastidx, lastidx + x.shape[0], x))

                for i in range(len(dataset_names)):
                    dset_to_append = dset[dataset_names[i]]

                    datasets_cat[i].append(dset_to_append[:])

                lastidx += old_shape
                dset.flush()
                count += 1

            except IOError:
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exc()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
                print
                "failed to open", cword

        for i in range(len(dataset_names)):
            datasets_cat[i] = np.concatenate(datasets_cat[i], axis=0)
            datasets_cat[i] = datasets_cat[i].transpose((1, 0))

        return images_data_cat, datasets_cat




        sensors_batch = []






        for i in range(len(images)):
            sensors_batch.append(np.zeros(
                (batch_size,
                 self._config.sensors_size[i][0],
                 self._config.sensors_size[i][1],
                 self._config.sensors_size[i][2] * self._number_frames_fused), dtype='uint8'
            ))

        for s in range(len(images)):
            count = 0

            for chosen_key in used_ids:

                count_seq = 0
                first_enter = True

                for i in range(self._number_frames_fused):
                    chosen_key = chosen_key + i * 3

                    for es, ee, x in images[s]:

                        if chosen_key >= es and chosen_key < ee:
                            """ We found the part of the data to open """
                            # print x[]
                            first_enter = False

                            pos_inside = chosen_key - es

                            # print 'el i'
                            # print chosen_key
                            # print pos_inside
                            # print x[chosen_key - es - 1 + 1:chosen_key - es + 1,:,:,:].shape

                            sensors_batch[s][count, :, :,
                            (i * 3):((i + 1) * 3)] = np.array(x[pos_inside, :, :, :])

                            # print sensors_batch[s][count].shape
                            #if not self._perform_sequential:
                            #img = Image.fromarray(sensors_batch[s][count])
                            #img.save('test' + str(self._current_position_on_dataset +count) + '_0_.jpg')
                count += 1

        return sensors_batch











        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample



    def _read_images2(self,images, batch_size, used_ids):
