import torch
import torch.nn.functional as F
import os
import pickle
import smplx
import numpy as np
from prior import MaxMixturePrior
import config


#  SMPLIfy 3D
class SMPLify3D():
    """Implementation of SMPLify, use 3D joints."""

    def __init__(self,
                 smplxmodel,
                 step_size=1e-2,
                 batch_size=1,
                 num_iters=100,
                 use_lbfgs=True,
                 use_collision=False,
                 joints_category="orig",
                 device=torch.device('cuda:0'),
                 ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=config.GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # collision part
        self.use_collision = use_collision
        if self.use_collision:
            self.part_segm_fn = config.Part_Seg_DIR
        
        # reLoad SMPL-X model
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # define the joint indices according to category
        self.joints_category = joints_category
        
        gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
        self.gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
        
        if joints_category=="orig":
            self.smpl_index = config.key_smpl_idx
            self.corr_index = config.key_smpl_idx
            self.joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
        elif joints_category=="h36m":
            self.smpl_index = config.h36m_smpl_idx
            self.corr_index = config.h36m_idx 
            self.joints_ind_category = [config.h36m_JOINT_MAP[joint] for joint in gt_joints]
        elif joints_category=="NTU":
            self.smpl_index = config.ntu_smpl_idx
            self.corr_index = config.ntu_idx 
            self.joints_ind_category = [config.NTU_JOINT_MAP[joint] for joint in gt_joints]
        elif joints_category=="CMU":
            self.smpl_index = config.cmu_smpl_idx
            self.corr_index = config.cmu_idx 
            self.joints_ind_category = [config.CMU_JOINT_MAP[joint] for joint in gt_joints]
        elif joints_category=="WU":
            self.smpl_index = config.wu_smpl_idx
            self.corr_index = config.wu_idx 
            self.joints_ind_category = [config.WU_JOINT_MAP[joint] for joint in gt_joints]
        else:
            self.smpl_index = None
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")
    
    @torch.no_grad()
    def guess_init_3d(self, model_joints, j3d):
        """Initialize the camera translation via triangle similarity, by using the torso joints        .
        :param model_joints: SMPL model with pre joints
        :param j3d: 25x3 array of Kinect Joints
        :returns: 3D vector corresponding to the estimated camera translation
        """
        # get the indexed four
        gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
        gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

        sum_init_t = (j3d[:, self.joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
        init_t = sum_init_t / 4.0
        return init_t
    
    # Gaussian
    def gmof(self, x, sigma):
        """
        Geman-McClure error function
        """
        x_squared = x ** 2
        sigma_squared = sigma ** 2
        return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    # angle prior
    def angle_prior(self, pose):
        """
        Angle prior that penalizes unnatural bending of the knees and elbows
        """
        # We subtract 3 because pose does not include the global rotation of the model
        return torch.exp(
            pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

    def perspective_projection(self, points, rotation, translation,
                               focal_length, camera_center):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points.shape[0]
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

        return projected_points[:, :, :-1]

    # camera fitting loss for 2d joints
    def camera_fitting_loss(self, model_joints, camera_t, camera_t_est, camera_center, 
                            joints_2d, joints_conf,
                            focal_length=5000, depth_loss_weight=100):
        """
        Loss function for camera optimization.
        """
        # Project model joints
        batch_size = model_joints.shape[0]
        rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
        projected_joints = self.perspective_projection(model_joints, rotation, camera_t,
                                                  focal_length, camera_center)

        # get the indexed four
        op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
        op_joints_ind = [config.JOINT_MAP[joint] for joint in op_joints]
        gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
        gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

        reprojection_error_op = (joints_2d[:, op_joints_ind] -
                                 projected_joints[:, op_joints_ind]) ** 2
        reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                                 projected_joints[:, gt_joints_ind]) ** 2

        # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
        # OpenPose joints are more reliable for this task, so we prefer to use them if possible
        is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
        reprojection_loss = (is_valid * reprojection_error_op + (1 - is_valid) * reprojection_error_gt).sum(dim=(1, 2))

        # Loss that penalizes deviation from depth estimate
        depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

        total_loss = reprojection_loss + depth_loss
        return total_loss.sum()
    
    # camera fitting loss for 3d joints
    def camera_fitting_loss_3d(self, model_joints, camera_t, camera_t_est,
                               j3d, depth_loss_weight=100):
        """
        Loss function for camera optimization.
        """
        for i in range(self.batch_size):
            model_joints[i,:,] = model_joints[i,:,] + camera_t[i,:]
        
        j3d_error_loss = (j3d[:, self.joints_ind_category] - model_joints[:, self.gt_joints_ind]) ** 2

        # Loss that penalizes deviation from depth estimate
        depth_loss = (depth_loss_weight ** 2) * (camera_t - camera_t_est) ** 2

        total_loss = j3d_error_loss.sum() + depth_loss
        return total_loss.sum()
        
    # body fitting with index EM
    def body_fitting_loss_em(self, body_pose, preserve_pose, betas, preserve_betas, camera_translation,
                             modelVerts, meshVerts, modelInd, meshInd, probInput, 
                             pose_prior,
                             sigma=100, pose_prior_weight=4.78,
                             shape_prior_weight=5.0, angle_prior_weight=15.2,
                             betas_loss_weight=1.0, pose_loss_weight=5.0,
                             correspond_weight=400.0):
        """
        Loss function for body fitting
        """
        batch_size = self.batch_size
        
        probInputM = torch.repeat_interleave(torch.reshape(probInput, (1, -1, 1) ), 3, dim=2)
        correspond_loss = correspond_weight * (probInputM *  self.gmof(modelVerts[:, modelInd] - meshVerts[:, meshInd], sigma) ).sum()
        # print(correspond_loss)
        
        # Pose prior loss
        pose_prior_loss = (pose_prior_weight ** 2) * self.pose_prior(body_pose, betas)
        # Angle prior for knees and elbows
        angle_prior_loss = (angle_prior_weight ** 2) * self.angle_prior(body_pose).sum(dim=-1)
        # Regularizer to prevent betas from taking large values
        shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

        betas_init_loss = (betas_loss_weight ** 2) * ((betas - preserve_betas) ** 2).sum(dim=-1)
        pose_init_loss = (pose_loss_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

        total_loss = correspond_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + betas_init_loss + pose_init_loss

        return total_loss.sum()
    
    # 2d body fitting loss
    def body_fitting_loss(self, body_pose, betas, model_joints, camera_t, camera_center,
                          joints_2d, joints_conf, pose_prior,
                          focal_length=5000, sigma=100, pose_prior_weight=4.78,
                          shape_prior_weight=5, angle_prior_weight=15.2,
                          output='sum'):
        """
        Loss function for body fitting
        """
        batch_size = self.batch_size
        rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)

        projected_joints = self.perspective_projection(model_joints, rotation, camera_t,
                                                  focal_length, camera_center)

        # Weighted robust reprojection error
        reprojection_error = self.gmof(projected_joints - joints_2d, sigma)
        reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

        # Pose prior loss
        pose_prior_loss = (pose_prior_weight ** 2) * self.pose_prior(body_pose, betas)

        # Angle prior for knees and elbows
        angle_prior_loss = (angle_prior_weight ** 2) * self.angle_prior(body_pose).sum(dim=-1)

        # Regularizer to prevent betas from taking large values
        shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

        total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

        if output == 'sum':
            return total_loss.sum()
        elif output == 'reprojection':
            return reprojection_loss

     # 3d body fitting loss
    def body_fitting_loss_3d(self, body_pose, preserve_pose,
                             betas, model_joints, camera_translation,
                             j3d, pose_prior,
                             joints3d_conf,
                             sigma=100, pose_prior_weight=4.78,
                             shape_prior_weight=5.0, angle_prior_weight=15.2,
                             joint_loss_weight=500.0,
                             pose_preserve_weight=0.0,
                             use_collision=False,
                             model_vertices=None, model_faces=None,
                             search_tree=None,  pen_distance=None,  filter_faces=None,
                             collision_loss_weight=1000
                             ):
        """
        Loss function for body fitting
        """
        batch_size = self.batch_size
        
        joint3d_error = self.gmof(model_joints + camera_translation.unsqueeze(1).repeat([1,j3d.shape[1],1]) - j3d, sigma)
        joint3d_loss_part = (joints3d_conf ** 2) * joint3d_error.sum(dim=-1)
        joint3d_loss = (joint_loss_weight ** 2) * joint3d_loss_part
        
        # Pose prior loss
        pose_prior_loss = (pose_prior_weight ** 2) * self.pose_prior(body_pose, betas)
        # Angle prior for knees and elbows
        angle_prior_loss = (angle_prior_weight ** 2) * self.angle_prior(body_pose).sum(dim=-1)
        # Regularizer to prevent betas from taking large values
        shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

        collision_loss = 0.0
        # Calculate the loss due to interpenetration
        if use_collision:
            triangles = torch.index_select(
                model_vertices, 1,
                model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = search_tree(triangles)

            # Remove unwanted collisions
            if filter_faces is not None:
                collision_idxs = filter_faces(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                collision_loss = torch.sum(collision_loss_weight * pen_distance(triangles, collision_idxs))
        
        pose_preserve_loss = (pose_preserve_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

        total_loss = joint3d_loss + (pose_prior_loss + angle_prior_loss + shape_prior_loss + collision_loss + pose_preserve_loss).unsqueeze(1).repeat([1, j3d.shape[1]])

        return total_loss.sum()

    
    # body fitting with know body position
    def body_fitting_loss_surface(self, body_pose, betas, preserve_betas, model_verts, camera_translation,
                                  points3d, indexes, pose_prior,
                                  sigma=100, pose_prior_weight=4.78*0.5,
                                  shape_prior_weight=5*0.5, angle_prior_weight=15.2,
                                  betas_loss_weight=10, joint_loss_weight=300):
        """
        Loss function for body fitting
        """
        batch_size = self.batch_size
        joint3d_loss = (joint_loss_weight ** 2) * self.gmof((model_verts[:, indexes] + camera_translation) - points3d, sigma).sum(dim=-1)

        # Pose prior loss
        pose_prior_loss = (pose_prior_weight ** 2) * self.pose_prior(body_pose, betas)
        # Angle prior for knees and elbows
        angle_prior_loss = (angle_prior_weight ** 2) * self.angle_prior(body_pose).sum(dim=-1)
        # Regularizer to prevent betas from taking large values
        shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

        betas_loss = (betas_loss_weight ** 2) * ((betas - preserve_betas) ** 2).sum(dim=-1)

        total_loss = joint3d_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + betas_loss

        return total_loss.sum()


    # depth render loss with preserved loss
    def body_render_depth_loss(self, body_pose, betas, preserve_betas, camera_translation,
                               depth_image, depth_rendered, pose_prior,
                               sigma=100, pose_prior_weight=4.78*0.5,
                               shape_prior_weight=5*0.5, angle_prior_weight=15.2,
                               betas_loss_weight=10):
        """
        Loss function for body fitting
        """
        batch_size = self.batch_size

        depth_loss = (200 ** 2) * F.l1_loss(depth_image, depth_rendered).sum(dim=-1)

        # Pose prior loss
        pose_prior_loss = (pose_prior_weight ** 2) * self.pose_prior(body_pose, betas)
        # Angle prior for knees and elbows
        angle_prior_loss = (angle_prior_weight ** 2) * self.angle_prior(body_pose).sum(dim=-1)
        # Regularizer to prevent betas from taking large values
        shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)
        # preserve predict betas
        betas_loss = (betas_loss_weight ** 2) * ((betas - preserve_betas) ** 2).sum(dim=-1)

        total_loss = depth_loss #+ pose_prior_loss + angle_prior_loss + shape_prior_loss + betas_loss

        return total_loss.sum()


    def __call__(self, init_pose, init_betas, init_cam_t, j3d, fit_beta=False, conf_3d=1.0):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        # add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None
        if self.use_collision:

            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            search_tree = BVH(max_collisions=8)

            pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                           sigma=0.5, point2plane=False, vectorized=True, penalize_outside=True)

            if self.part_segm_fn:
                # Read the part segmentation
                part_segm_fn = os.path.expandvars(self.part_segm_fn)
                with open(part_segm_fn, 'rb') as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file,  encoding='latin1')
                faces_segm = face_segm_data['segm']
                faces_parents = face_segm_data['parents']
                # Create the module used to filter invalid collision pairs
                filter_faces = FilterFaces(
                    faces_segm=faces_segm, faces_parents=faces_parents,
                    ign_part_pairs=None).to(device=self.device)
                    
                    
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas)
        model_joints = smpl_output.joints

        init_cam_t = self.guess_init_3d(model_joints, j3d).detach()
        camera_translation = init_cam_t.clone()
        
        preserve_pose = init_pose[:, 3:].detach().clone()
        
        # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.num_iters,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(10):
                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints

                    loss = self.camera_fitting_loss_3d(model_joints, camera_translation, init_cam_t, j3d)
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(20):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints

                loss = self.camera_fitting_loss_3d(model_joints[:, self.smpl_index], camera_translation, init_cam_t,  j3d[:, self.corr_index])
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        # --- if we use the sequence, fix the shape
        if fit_beta:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.num_iters, lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = self.body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                                j3d[:, self.corr_index], self.pose_prior,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                pose_preserve_weight=5.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = self.body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                                j3d[:, self.corr_index], self.pose_prior,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                use_collision=self.use_collision, 
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree,  pen_distance=pen_distance,  filter_faces=filter_faces)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices

            final_loss = self.body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                                  j3d[:, self.corr_index], self.pose_prior,
                                                  joints3d_conf=conf_3d,
                                                  joint_loss_weight=600.0,
                                                  use_collision=self.use_collision, model_vertices=model_vertices, model_faces=self.model_faces,
                                                  search_tree=search_tree,  pen_distance=pen_distance,  filter_faces=filter_faces)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()
        camera_translation = camera_translation.detach()

        return vertices, joints, pose, betas, camera_translation, final_loss
