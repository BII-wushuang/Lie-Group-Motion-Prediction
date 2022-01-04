# SMPL Fitting 流程

1. 骨架定义可参考 H3.6m Kinematics notebook
   - 把需要跑Mesh Fitting的3D坐标（npy格式）放入Results文件夹
   - 注意关节点的对应关系
   - 另外确保x、y、z轴的顺序（y-轴对应人体直立）
2. 接下来跑fit_smpl.py进行SMPL参数的拟合
   - 这一步输出为Rendered/dataset.pkl
   - 可以在fit_smpl.py 76行修改骨架joints_category
   - 如果需要重新定义骨架关节点，需要在config.py内修改，并在 simplify.py 74行后定义
3. Visualization notebook 进行可视化
   - 输出在Videos/dataset/