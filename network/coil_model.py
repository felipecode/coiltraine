from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class CoILModel(nn.Module):

    def __init__(self, MODEL_DEFINITION):

        super(CoILModel, self).__init__()
        # TODO: MAKE THE model
        """" ------------------ IMGAE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization
        """ conv1 + batch normalization + dropout + relu """
        self.im_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2)
        init.xavier_uniform_(self.im_conv1.weight)
        init.constant_(self.im_conv1.bias, 0.1)
        self.im_conv1_bn = nn.BatchNorm2d(32)
        self.im_conv1_drop = nn.Dropout2d(p=0.2)

        """ conv2 + batch normalization + dropout + relu """
        self.im_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        init.xavier_uniform_(self.im_conv2.weight).cuda()
        init.constant_(self.im_conv2.bias, 0.1)
        self.im_conv2_bn = nn.BatchNorm2d(32)
        self.im_conv2_drop = nn.Dropout2d(p=0.2)

        """ conv3 + batch normalization + dropout + relu """
        self.im_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        init.xavier_uniform_(self.im_conv3.weight)
        init.constant_(self.im_conv3.bias, 0.1)
        self.im_conv3_bn = nn.BatchNorm2d(64)
        self.im_conv3_drop = nn.Dropout2d(p=0.2)

        """ conv4 + batch normalization + dropout + relu """
        self.im_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        init.xavier_uniform_(self.im_conv4.weight)
        init.constant_(self.im_conv4.bias, 0.1)
        self.im_conv4_bn = nn.BatchNorm2d(64)
        self.im_conv4_drop = nn.Dropout2d(p=0.2)

        """ conv5 + batch normalization + dropout + relu """
        self.im_conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        init.xavier_uniform_(self.im_conv5.weight)
        init.constant_(self.im_conv5.bias, 0.1)
        self.im_conv5_bn = nn.BatchNorm2d(128)
        self.im_conv5_drop = nn.Dropout2d(p=0.2)

        """ conv6 + batch normalization + dropout + relu """
        self.im_conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        init.xavier_uniform_(self.im_conv6.weight)
        init.constant_(self.im_conv6.bias, 0.1)
        self.im_conv6_bn = nn.BatchNorm2d(128)
        self.im_conv6_drop = nn.Dropout2d(p=0.2)

        """ conv7 + batch normalization + dropout + relu """
        self.im_conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        init.xavier_uniform_(self.im_conv7.weight)
        init.constant_(self.im_conv7.bias, 0.1)
        self.im_conv7_bn = nn.BatchNorm2d(256)
        self.im_conv7_drop = nn.Dropout2d(p=0.2)

        """ conv8 + batch normalization + dropout + relu """
        self.im_conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        init.xavier_uniform_(self.im_conv8.weight)
        init.constant_(self.im_conv8.bias, 0.1)
        self.im_conv8_bn = nn.BatchNorm2d(256)
        self.im_conv8_drop = nn.Dropout2d(p=0.2)

        """ fc1 + dropout + relu """
        self.im_fc1 = nn.Linear(8192, 512)
        init.xavier_uniform_(self.im_fc1.weight)
        init.constant_(self.im_fc1.bias, 0.1)
        self.im_fc1_drop = nn.Dropout2d(p=0.5)

        """ fc1 + dropout + relu """
        self.im_fc2 = nn.Linear(512, 512)
        init.xavier_uniform_(self.im_fc2.weight)
        init.constant_(self.im_fc2.bias, 0.1)
        self.im_fc2_drop = nn.Dropout2d(p=0.5)

        """" ---------------------- SPEED MODULE ----------------------- """
        self.sp_fc1 = nn.Linear(1, 128)
        init.xavier_uniform_(self.sp_fc1.weight)
        init.constant_(self.sp_fc1.bias, 0.1)
        self.sp_fc1_drop = nn.Dropout2d(p=0.5)

        self.sp_fc2 = nn.Linear(128, 128)
        init.xavier_uniform_(self.sp_fc2.weight)
        init.constant_(self.sp_fc2.bias, 0.1)
        self.sp_fc2_drop = nn.Dropout2d(p=0.5)

        """ ---------------------- J MODULE ---------------------------- """
        self.j_fc1 = nn.Linear(640, 512)
        init.xavier_uniform_(self.j_fc1.weight)
        init.constant_(self.j_fc1.bias, 0.1)
        self.j_fc1_drop = nn.Dropout2d(p=0.5)

        """ ---------------------- BRANCHING MODULE --------------------- """
        self.branch_fc1 = nn.Linear(512, 256)
        init.xavier_uniform_(self.branch_fc1.weight)
        init.constant_(self.branch_fc1.bias, 0.1)
        self.branch_fc1_drop = nn.Dropout2d(p=0.5)

        self.branch_fc2 = nn.Linear(256, 256)
        init.xavier_uniform_(self.branch_fc2.weight)
        init.constant_(self.branch_fc2.bias, 0.1)
        self.branch_fc2_drop = nn.Dropout2d(p=0.5)

        # define output fc for branch 1,2,3,4
        self.branch_fc_3out = nn.Linear(256, 3)
        init.xavier_uniform_(self.branch_fc_3out.weight)
        init.constant_(self.branch_fc_3out.bias, 0.1)

        # define output fc for speed branch
        self.branch_fc_1out = nn.Linear(256, 1)
        init.xavier_uniform_(self.branch_fc_1out.weight)
        init.constant_(self.branch_fc_1out.bias, 0.1)



    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x, labels):
        # get the speeds and the hight level control commands from measurement labels
        speed = labels[:, 10, :]

        # TODO: TRACK NANS OUTPUTS
        # TODO: Maybe change the name
        coil_logger.add_message('Model', {
            "Iteration": 765,
            "Output": [1.0, 12.3, 124.29]
        }
                                )
        branches = []
        """ conv1 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv1_drop(self.im_conv1_bn(self.im_conv1(x))))

        """ conv2 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv2_drop(self.im_conv2_bn(self.im_conv2(x))))

        """ conv3 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv3_drop(self.im_conv3_bn(self.im_conv3(x))))

        """ conv4 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv4_drop(self.im_conv4_bn(self.im_conv4(x))))

        """ conv5 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv5_drop(self.im_conv5_bn(self.im_conv5(x))))

        """ conv6 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv6_drop(self.im_conv6_bn(self.im_conv6(x))))

        """ conv7 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv7_drop(self.im_conv7_bn(self.im_conv7(x))))

        """ conv8 + batch normalization + dropout + relu """
        x = F.relu(self.im_conv8_drop(self.im_conv8_bn(self.im_conv8(x))))

        x = x.view(-1, self.num_flat_features(x))

        """ fc1 + dropout + relu """
        x = F.relu(self.im_fc1_drop(self.im_fc1(x)))

        """ fc2 + dropout """
        x = self.im_fc2_drop(self.im_fc2(x))

        """ speed fc """
        speed = F.relu(self.sp_fc1_drop(self.sp_fc1(speed)))
        speed = self.sp_fc2_drop(self.sp_fc2(speed))

        """ j fc """
        j = torch.cat((x, speed), 1)
        j = F.relu(self.j_fc1_drop(self.j_fc1(j)))

        """ Start BRANCHING """
        branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                         ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]]

        for i in range(0, len(branch_config)):
            if branch_config[i][0] == "Speed":
                # only use the image as input to speed prediction
                branch_output = F.relu(self.branch_fc1_drop(self.branch_fc1(x)))
                branch_output = self.branch_fc2_drop(self.branch_fc2(branch_output))
                branch_results = self.branch_fc_1out(branch_output)
            else:
                branch_output = F.relu(self.branch_fc1_drop(self.branch_fc1(j)))
                branch_output = self.branch_fc2_drop(self.branch_fc2(branch_output))
                branch_results = self.branch_fc_3out(branch_output)
            branches.append(branch_results)
        return branches

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })



        # TODO: implement



