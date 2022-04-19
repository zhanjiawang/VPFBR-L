#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <eigen3/Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

typedef struct voxelnode
{
	float centry_x;
	float centry_y;
	float centry_z;
	float normal_x;
	float normal_y;
	float normal_z;
	int voxel_point_size;
	bool is_allocate;
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_ptr;
}voxelnode;

typedef struct facenode
{
	float average_centry_x;
	float average_centry_y;
	float average_centry_z;
	float average_normal_x;
	float average_normal_y;
	float average_normal_z;
	float face_point_size;
	bool is_allocate;
	std::vector<voxelnode> voxelgrothnode;
}facenode;

typedef struct describ
{
	std::vector<float> theta;
	std::vector<float> radiu;
	std::vector<bool> maindir;
	std::vector<int> size;
	std::vector<std::vector<float> > distribute;
}describ;

typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerT;

//point cloud down sampling
float LeafSize=0.2;

//parameter for plane feature extraction and fusion
//l and k
float parameter_l1=1.0;
float parameter_l2=1.0;
float parameter_l3=1.0;
float parameter_k1=8.0;
float parameter_k2=4.0;
float parameter_k3=2.0;
//threshold of normal vector
float normal_vector_threshold1=5.0;
float normal_vector_threshold2=10.0;
float normal_vector_threshold3=15.0;
//voxel size
float face_voxel_size=1.0;
//if the point in voxel is less than this value, it will not be processed
float voxel_point_threshold=10;
//threshold of plane curvature
float curvature_threshold=0.08;
//maximum plane distance
float max_plane_distance=5.0;
//select the size of planes
float select_plane_size=5;

//Combination plane features and augmented descriptors parameters
//Select main directions number
int miannum=4;
//length of augmented descriptors
int alonglength=100;
//interval of the block
float alongresolution=0.5;
//parameters of angle difference weight and the distance difference weight
float angelthreashold=5.0;
float radiuthreashold=5.0;
float downto=0.75;
//feature similarity weight of combination plane features
float weight_combination_plane_features=0.25;
//feature similarity weight of augmented descriptors
float weight_augmented_descriptors=0.5;

float compute_angel(float x,float y)
{
	float angel=0;
	if(x==0 && y==0)
	{
		angel=0;
	}else if(x==1 && y==0)
	{
		angel=0;
	}else if(x>0 && y==0)
	{
		angel=0;
	}else if(x>0 && y>0)
	{
		angel=fabs((180*atan2(x,y)/M_PI)-90);
	}else if(x==0 && y==1)
	{
		angel=90;
	}else if(x<0 && y>0)
	{
		angel=fabs((180*atan2(x,y)/M_PI))+90;
	}else if(x==-1 && y==0)
	{
		angel=180;
	}else if(x<0 && y<0)
	{
		angel=fabs((180*atan2(x,y)/M_PI))+90;
	}else if(x==0 && y==-1)
	{
		angel=270;
	}else
	{
		angel=fabs((180*atan2(x,y)/M_PI)-180)+270;
	}
	return angel;
}

float compute_normal_angel(float normal_x1,float normal_y1,float normal_z1,float normal_x2,float normal_y2,float normal_z2)
{
	Eigen::Vector3d n1=Eigen::Vector3d(normal_x1,normal_y1,normal_z1);
	Eigen::Vector3d n2=Eigen::Vector3d(normal_x2,normal_y2,normal_z2);
    float n1n3=n1.transpose()*n2;
    float cos_theta=n1n3/((n1.norm())*(n2.norm()));
    float theta=acos(cos_theta)*180/M_PI;
	return theta;
}

bool compare_normal(float normal_x1,float normal_y1,float normal_z1,float normal_x2,float normal_y2,float normal_z2,float normal_vector_threshold)
{
	float theta=compute_normal_angel(normal_x1,normal_y1,normal_z1,normal_x2,normal_y2,normal_z2);
	if(theta>normal_vector_threshold)
	{
		return false;
	}else
	{
		return true;
	}
}

bool compare_plane(float normal_x1,float normal_y1,float normal_z1,float centry_x1,float centry_y1,float centry_z1,float normal_x2,float normal_y2,float normal_z2,float centry_x2,float centry_y2,float centry_z2,float parameter_l,float parameter_k)
{
	Eigen::Vector3d n1=Eigen::Vector3d(normal_x1,normal_y1,normal_z1);
	Eigen::Vector3d n2=Eigen::Vector3d(normal_x2,normal_y2,normal_z2);
	float vectorlength= sqrt((centry_x1-centry_x2)*(centry_x1-centry_x2)+(centry_y1-centry_y2)*(centry_y1-centry_y2)+(centry_z1-centry_z2)*(centry_z1-centry_z2));
	if(vectorlength>max_plane_distance)
	{
		vectorlength=max_plane_distance;
	}	
	Eigen::Vector3d n3=Eigen::Vector3d((centry_x1-centry_x2)/vectorlength,(centry_y1-centry_y2)/vectorlength,(centry_z1-centry_z2)/vectorlength);
	float n1n3=fabs(n1.transpose()*n3);
	float n2n3=fabs(n2.transpose()*n3);
	float co_plane_threash=parameter_l/(parameter_k*vectorlength+1);
	if(n1n3<co_plane_threash && n2n3<co_plane_threash)
	{
		return true;
	}else
	{
		return false;
	}
}

void range_face(std::vector<facenode> &face_vecter)
{
	for(auto it1=face_vecter.begin();it1!=face_vecter.end();it1++)
	{
		if(it1!=(face_vecter.end()-1))
		{
			for(auto it2=it1+1;it2!=face_vecter.end();it2++)
			{
				if((*it1).voxelgrothnode.size()<(*it2).voxelgrothnode.size())
				{
					facenode facenodetemp;
					facenodetemp=(*it1);
					(*it1)=(*it2);
					(*it2)=facenodetemp;
				}
			}
		}
	}
}

void select_face(std::vector<facenode> &face_vecter)
{
	float chose_threshold=0.1;
	std::vector<facenode> select_face_vecter;
	for(auto it1=face_vecter.begin();it1!=face_vecter.end();it1++)
	{
		if(fabs((*it1).average_normal_z)<chose_threshold)
		{
			select_face_vecter.push_back((*it1));
		}
	}
	face_vecter.swap(select_face_vecter);
}

void face_extrate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src,std::vector<facenode> &face_vecter)
{	
	Eigen::Matrix<float, 4, 1> cloud_src_centroid; 
	pcl::compute3DCentroid(*cloud_src, cloud_src_centroid);
	float resolution = face_voxel_size;
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
	octree.setInputCloud(cloud_src);
	octree.addPointsFromInputCloud();
	int depth = octree.getTreeDepth();
	pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector vec;
	octree.getOccupiedVoxelCenters(vec);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud< pcl::Normal>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid (new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<voxelnode> voxel_vector;
	for(int i=0;i<vec.size();i++)
	{
		std::vector<int> pointIdxVec;
		if (octree.voxelSearch ((*(vec.begin()+i)), pointIdxVec))
		{
			if(pointIdxVec.size ()>voxel_point_threshold)
			{
				pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
				Eigen::Matrix<float, 4, 1> centroid;
				pcl::compute3DCentroid((*cloud_src),pointIdxVec,centroid);
				float nx;
				float ny;
				float nz;
				float curvature;
                ne_src.computePointNormal((*cloud_src),pointIdxVec,nx,ny,nz,curvature);
				if(curvature<curvature_threshold)
				{
					voxelnode voxelnode_temp;
					voxelnode_temp.centry_x=centroid[0];
					voxelnode_temp.centry_y=centroid[1];
					voxelnode_temp.centry_z=centroid[2];
					voxelnode_temp.voxel_point_size=pointIdxVec.size ();
					Eigen::Vector3f to_centry=Eigen::Vector3f((centroid[0]-cloud_src_centroid[0]),(centroid[1]-cloud_src_centroid[1]),(centroid[2]-cloud_src_centroid[2]));
					Eigen::Vector3f normal_vector=Eigen::Vector3f(nx,ny,nz);
					if((to_centry.dot(normal_vector))<0)
					{
						voxelnode_temp.normal_x=nx;
						voxelnode_temp.normal_y=ny;
						voxelnode_temp.normal_z=nz;
					}else
					{
						voxelnode_temp.normal_x=-nx;
						voxelnode_temp.normal_y=-ny;
						voxelnode_temp.normal_z=-nz;						
					}
					voxelnode_temp.is_allocate=false;
					pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud_ptr_temp(new pcl::PointCloud<pcl::PointXYZ>);
					for(auto it=pointIdxVec.begin();it!=pointIdxVec.end();it++)
					{
						(*voxel_cloud_ptr_temp).push_back((*cloud_src)[*it]);
					}
					voxelnode_temp.voxel_cloud_ptr=voxel_cloud_ptr_temp;
					voxel_vector.push_back(voxelnode_temp);
				}
			}			
		}
	}
		
    std::vector<facenode> voxel_vector_groth;
	for(auto it1=voxel_vector.begin();it1!=voxel_vector.end();it1++)
	{
		if((*it1).is_allocate==false)
		{
			facenode facenode_temp;
			(*it1).is_allocate=true;
			facenode_temp.voxelgrothnode.push_back(*it1);
			facenode_temp.face_point_size=(*it1).voxel_point_size;
			facenode_temp.average_normal_x=(*it1).normal_x;
			facenode_temp.average_normal_y=(*it1).normal_y;
			facenode_temp.average_normal_z=(*it1).normal_z;	
			facenode_temp.average_centry_x=(*it1).centry_x;
			facenode_temp.average_centry_y=(*it1).centry_y;
			facenode_temp.average_centry_z=(*it1).centry_z;
			for(auto it2=voxel_vector.begin();it2!=voxel_vector.end();it2++)
			{
				if((*it2).is_allocate==false)
				{
					bool is_same=true;
					bool is_coplane=true;
					is_same=compare_normal(facenode_temp.average_normal_x,facenode_temp.average_normal_y,facenode_temp.average_normal_z,(*it2).normal_x,(*it2).normal_y,(*it2).normal_z,normal_vector_threshold1);
					is_coplane=compare_plane(facenode_temp.average_normal_x,facenode_temp.average_normal_y,facenode_temp.average_normal_z,facenode_temp.average_centry_x,facenode_temp.average_centry_y,facenode_temp.average_centry_z,(*it2).normal_x,(*it2).normal_y,(*it2).normal_z,(*it2).centry_x,(*it2).centry_y,(*it2).centry_z,parameter_l1,parameter_k1);
					if(is_same==true && is_coplane==true)
					{
						facenode_temp.voxelgrothnode.push_back(*it2);
						(*it2).is_allocate=true;
						float sum_voxel_point_size=0;
						float average_centry_x=0;
						float average_centry_y=0;
						float average_centry_z=0;
						float average_normal_x=0;
						float average_normal_y=0;
						float average_normal_z=0;			
						for(auto it3=(facenode_temp.voxelgrothnode).begin();it3!=(facenode_temp.voxelgrothnode).end();it3++)
						{
							sum_voxel_point_size=sum_voxel_point_size+((*it3).voxel_point_size);
							average_centry_x=average_centry_x+((*it3).centry_x)*((*it3).voxel_point_size);
							average_centry_y=average_centry_y+((*it3).centry_y)*((*it3).voxel_point_size);
							average_centry_z=average_centry_z+((*it3).centry_z)*((*it3).voxel_point_size);
							average_normal_x=average_normal_x+((*it3).normal_x)*((*it3).voxel_point_size);
							average_normal_y=average_normal_y+((*it3).normal_y)*((*it3).voxel_point_size);
							average_normal_z=average_normal_z+((*it3).normal_z)*((*it3).voxel_point_size);	
						}
						facenode_temp.face_point_size=sum_voxel_point_size;
						facenode_temp.average_centry_x=average_centry_x/sum_voxel_point_size;
						facenode_temp.average_centry_y=average_centry_y/sum_voxel_point_size;
						facenode_temp.average_centry_z=average_centry_z/sum_voxel_point_size;
						facenode_temp.average_normal_x=average_normal_x/sum_voxel_point_size;
						facenode_temp.average_normal_y=average_normal_y/sum_voxel_point_size;
						facenode_temp.average_normal_z=average_normal_z/sum_voxel_point_size;
					}					
				}
			}	
			facenode_temp.is_allocate=false;
			voxel_vector_groth.push_back(facenode_temp);
		}  
	}

    for(auto it1=voxel_vector_groth.begin();it1!=voxel_vector_groth.end();it1++)
	{	
		if((*it1).is_allocate==false)
		{
			bool newadd=true;
			while(newadd==true)
			{
				newadd=false;
				for(auto it2=voxel_vector_groth.begin();it2!=voxel_vector_groth.end();it2++)
				{
					if((it2)!=(it1) && (*it2).is_allocate==false)
					{
						bool is_same=true;
						bool is_coplane=true;
						is_same=compare_normal((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z,(*it2).average_normal_x,(*it2).average_normal_y,(*it2).average_normal_z,normal_vector_threshold2);
						is_coplane=compare_plane((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z,(*it1).average_centry_x,(*it1).average_centry_y,(*it1).average_centry_z,(*it2).average_normal_x,(*it2).average_normal_y,(*it2).average_normal_z,(*it2).average_centry_x,(*it2).average_centry_y,(*it2).average_centry_z,parameter_l2,parameter_k2);
						if(is_same==true && is_coplane==true)
						{
							newadd=true;
							(*it2).is_allocate=true;
							for(auto it4=((*it2).voxelgrothnode).begin();it4!=((*it2).voxelgrothnode).end();it4++)
							{
								((*it1).voxelgrothnode).push_back(*it4);
							}
							float sum_voxel_point_size=0;
							float average_centry_x=0;
							float average_centry_y=0;
							float average_centry_z=0;
							float average_normal_x=0;
							float average_normal_y=0;
							float average_normal_z=0;			
							for(auto it3=((*it1).voxelgrothnode).begin();it3!=((*it1).voxelgrothnode).end();it3++)
							{
								sum_voxel_point_size=sum_voxel_point_size+((*it3).voxel_point_size);
								average_centry_x=average_centry_x+((*it3).centry_x)*((*it3).voxel_point_size);
								average_centry_y=average_centry_y+((*it3).centry_y)*((*it3).voxel_point_size);
								average_centry_z=average_centry_z+((*it3).centry_z)*((*it3).voxel_point_size);
								average_normal_x=average_normal_x+((*it3).normal_x)*((*it3).voxel_point_size);
								average_normal_y=average_normal_y+((*it3).normal_y)*((*it3).voxel_point_size);
								average_normal_z=average_normal_z+((*it3).normal_z)*((*it3).voxel_point_size);	
							}
							(*it1).face_point_size=sum_voxel_point_size;
							(*it1).average_centry_x=average_centry_x/sum_voxel_point_size;
							(*it1).average_centry_y=average_centry_y/sum_voxel_point_size;
							(*it1).average_centry_z=average_centry_z/sum_voxel_point_size;
							(*it1).average_normal_x=average_normal_x/sum_voxel_point_size;
							(*it1).average_normal_y=average_normal_y/sum_voxel_point_size;
							(*it1).average_normal_z=average_normal_z/sum_voxel_point_size;
						}					
					}
				}
			}
		}
	}

	range_face(voxel_vector_groth);
	std::vector<facenode> face_vecter_chose;
	for(auto it1=voxel_vector_groth.begin();it1!=voxel_vector_groth.end();it1++)
	{
		if((*it1).voxelgrothnode.size()>=select_plane_size)
		{
			bool canpush=true;
			for(auto it2=face_vecter_chose.begin();it2!=face_vecter_chose.end();it2++)
			{
				bool is_same=true;
				bool is_coplane=true;
				is_same=compare_normal((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z,(*it2).average_normal_x,(*it2).average_normal_y,(*it2).average_normal_z,normal_vector_threshold3);
				is_coplane=compare_plane((*it1).average_normal_x,(*it1).average_normal_y,(*it1).average_normal_z,(*it1).average_centry_x,(*it1).average_centry_y,(*it1).average_centry_z,(*it2).average_normal_x,(*it2).average_normal_y,(*it2).average_normal_z,(*it2).average_centry_x,(*it2).average_centry_y,(*it2).average_centry_z,parameter_l3,parameter_k3);
				if(is_same==true && is_coplane==true)
				{
					canpush=false;
				}				
			}
			if(canpush==true)
			{
				face_vecter_chose.push_back((*it1));
			}
		}	
	}
	face_vecter.swap(face_vecter_chose);
}

void creat_describ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,std::vector<facenode> &face_vecter,describ &des)
{
	float face_sum_x=0;
	float face_sum_y=0;
	float face_sum_z=0;
	float face_sum_point=0;		
	for(auto it1=face_vecter.begin();it1!=face_vecter.end();it1++)
	{
		face_sum_x=face_sum_x+(*it1).average_centry_x*(*it1).face_point_size;
		face_sum_y=face_sum_y+(*it1).average_centry_y*(*it1).face_point_size;
		face_sum_z=face_sum_z+(*it1).average_centry_z*(*it1).face_point_size;
		face_sum_point=face_sum_point+(*it1).face_point_size;
	}
	float face_centry_x=face_sum_x/face_sum_point;
	float face_centry_y=face_sum_y/face_sum_point;
	float face_centry_z=face_sum_z/face_sum_point;
	select_face(face_vecter);

	int index=0;
	for(auto it1=face_vecter.begin();it1!=face_vecter.end();it1++)
	{
		float theta=compute_angel((*it1).average_normal_x,(*it1).average_normal_y);
		Eigen::Vector3f cs=Eigen::Vector3f((*it1).average_centry_x-face_centry_x,(*it1).average_centry_y-face_centry_y,0);
		Eigen::Vector3f ns=Eigen::Vector3f((*it1).average_normal_x,(*it1).average_normal_y,0);
		ns.normalize();
		float radiu=fabs(cs.dot(ns));
		des.theta.push_back(theta);
		des.radiu.push_back(radiu);
		des.size.push_back((*it1).face_point_size);
		if(index<miannum)
		{
			des.maindir.push_back(true);
			Eigen::Vector3f nt=Eigen::Vector3f(1,0,0);
			Eigen::Vector3f r=ns.cross(nt);
			r.normalize(); 
			Eigen::Matrix3f rx = Eigen::Matrix3f::Identity();
			rx(0,0)=0;
			rx(0,1)=-r[2];
			rx(0,2)=r[1];
			rx(1,0)=r[2];
			rx(1,1)=0;
			rx(1,2)=-r[0];
			rx(2,0)=-r[1];
			rx(2,1)=r[0];
			rx(2,2)=0;	
			Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
			Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
			float cos_theta=nt.dot(ns);
			float sin_theta=nt.dot(r.cross(ns));
			Eigen::Matrix3f rrt = r*(r.transpose());
			R=cos_theta*I+(1-cos_theta)*rrt+sin_theta*rx;
			Eigen::Vector3f cz=Eigen::Vector3f((*it1).average_centry_x,(*it1).average_centry_y,0);
			cz=R*cz;
			Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
			T(0,0)=R(0,0);
			T(0,1)=R(0,1);
			T(0,2)=R(0,2);
			T(0,3)=-cz[0];
			T(1,0)=R(1,0);
			T(1,1)=R(1,1);
			T(1,2)=R(1,2);
			T(1,3)=-cz[1];
			T(2,0)=R(2,0);
			T(2,1)=R(2,1);
			T(2,2)=R(2,2);
			Eigen::Quaternionf q(R);
			pcl::transformPointCloud (*cloud, *cloud, T);	
            std::vector<float> distribute(alonglength*2);
			int pointindex=0;
			for (const auto &point: *cloud)
			{				
				if(point.x>0)
				{
					int index=point.x/alongresolution;
					if(index>=0 && index<alonglength)
					{
						distribute[index+alonglength]=distribute[index+alonglength]+point.z;
					}
				}else if(point.x<0)
				{
					int index=fabs(point.x)/alongresolution;
					if(index>=0 && index<alonglength)
					{
						distribute[alonglength-1-index]=distribute[alonglength-1-index]+point.z;
					}
				}
				pointindex++;	
			}			
			des.distribute.push_back(distribute);			
			pcl::transformPointCloud (*cloud, *cloud, T.inverse());						
		}else
		{
			des.maindir.push_back(false);
		}
		index++;
	}
}

float angel_subtract(float minuend,float subtractor)
{
	float subtract=minuend-subtractor;
	if(subtract>=0)
	{
		return subtract;
	}else
	{
		return 360+subtract;
	}
	
}

float angel_distance(float angels,float angelt)
{
	float angelmax=angels>angelt?angels:angelt;
	float angelmin=angels<angelt?angels:angelt;
	float distance=angelmax-angelmin;
	if(distance>180)
	{
		distance=360-distance;
	}
	return distance;
}

float compare_describ(describ &dess,describ &dest)
{
	float sizes=0;
	float sizet=0;
	float mains=0;
	float maint=0;
	for(int i=0;i<dess.size.size();i++)
	{
		sizes=sizes+dess.size[i];
		if(dess.maindir[i]==true)
		{
			mains++;
		}
	}
	for(int i=0;i<dest.size.size();i++)
	{
		sizet=sizet+dest.size[i];
		if(dest.maindir[i]==true)
		{
			maint++;
		}		
	}
	float angelk=((1/downto)-1)/angelthreashold;
	float radiuk=((1/downto)-1)/radiuthreashold;

	float bestscore=0;
	for(int i=0;i<dess.maindir.size();i++)
	{
		if(dess.maindir[i]==true)
		{
			for(int j=0;j<dest.maindir.size();j++)
			{
				if(dest.maindir[j]==true)
				{
					std::vector<float> angels;
					std::vector<float> angelt;
					for(int k=0;k<dess.theta.size();k++)
					{
						float angel=angel_subtract(dess.theta[k],dess.theta[i]);
						angels.push_back(angel);
					}	
					for(int l=0;l<dest.theta.size();l++)
					{
						float angel=angel_subtract(dest.theta[l],dest.theta[i]);
						angelt.push_back(angel);
					}

					float scorefacefeaturet=0;
					for(int k=0;k<angelt.size();k++)
					{
						float bestcandatescore=0;
						std::vector<int> candidate;
						std::vector<float> candidateangledis;
						std::vector<float> candidateradiudis;
						for(int l=0;l<angels.size();l++)
						{
							float angeldis=angel_distance(angelt[k],angels[l]);
							float radiudis=fabs(dest.radiu[k]-dess.radiu[l]);
							if(angeldis<angelthreashold && radiudis<radiuthreashold)
							{
								candidate.push_back(l);
								candidateangledis.push_back(angeldis);
								candidateradiudis.push_back(radiudis);
							}
						}
						for(int m=0;m<candidate.size();m++)
						{
							float minsize=dest.size[k]<dess.size[candidate[m]]?dest.size[k]:dess.size[candidate[m]];
							float maxsize=dest.size[k]>dess.size[candidate[m]]?dest.size[k]:dess.size[candidate[m]];
							float candidatescore=(1/(radiuk*candidateradiudis[m]+1))*(1/(angelk*candidateangledis[m]+1))*(minsize/maxsize)*((minsize+maxsize)/(sizes+sizet));
							if(candidatescore>bestcandatescore)
							{
								bestcandatescore=candidatescore;
							}
						}
						scorefacefeaturet=scorefacefeaturet+bestcandatescore;						
					}

					float scorefacefeatures=0;
					float scoredistirbution=0;					
					for(int k=0;k<angels.size();k++)
					{
						float bestcandatescore=0;
						float bestsimilarscore=0;
						std::vector<int> candidate;
						std::vector<float> candidateangledis;
						std::vector<float> candidateradiudis;
						for(int l=0;l<angelt.size();l++)
						{
							float angeldis=angel_distance(angels[k],angelt[l]);
							float radiudis=fabs(dess.radiu[k]-dest.radiu[l]);
							if(angeldis<angelthreashold && radiudis<radiuthreashold)
							{
								candidate.push_back(l);
								candidateangledis.push_back(angeldis);
								candidateradiudis.push_back(radiudis);
							}
						}
						for(int m=0;m<candidate.size();m++)
						{
							float similarscore=0;
							if(dess.maindir[k]==true && dest.maindir[candidate[m]]==true)
							{
								for(int n=0;n<alonglength*2;n++)
								{
									float nums=(dess.distribute[k])[n];
									float numt=(dest.distribute[candidate[m]])[n];
									float minnum=nums<numt?nums:numt;
									float maxnum=nums>numt?nums:numt;									
									if(nums>0 && numt>0)
									{
										similarscore=similarscore+(minnum/maxnum)/(alonglength*2);
									}else if(nums<0 && numt<0)
									{
										similarscore=similarscore+(maxnum/minnum)/(alonglength*2);										
									}else
									{
										similarscore=similarscore;
									}
								}
							}
							if(similarscore>bestsimilarscore)
							{
								bestsimilarscore=similarscore;
							}							
							float minsize=dess.size[k]<dest.size[candidate[m]]?dess.size[k]:dest.size[candidate[m]];
							float maxsize=dess.size[k]>dest.size[candidate[m]]?dess.size[k]:dest.size[candidate[m]];
							float candidatescore=(1/(radiuk*candidateradiudis[m]+1))*(1/(angelk*candidateangledis[m]+1))*(minsize/maxsize)*((minsize+maxsize)/(sizes+sizet));
							if(candidatescore>bestcandatescore)
							{
								bestcandatescore=candidatescore;
							}
						}
						scorefacefeatures=scorefacefeatures+bestcandatescore;
						scoredistirbution=scoredistirbution+bestsimilarscore;					
					}
					float score=scorefacefeaturet*weight_combination_plane_features+scorefacefeatures*weight_combination_plane_features+scoredistirbution*weight_augmented_descriptors;					
					if(score>bestscore)	
					{
						bestscore=score;
					}								
				}
			}			
		}
	}
	return bestscore;
}

int main (int argc, char** argv)
{ 
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *source) == -1)
	{
		PCL_ERROR("Couldn't read file \n");
		return (-1);
	}
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_source;
	voxel_grid_source.setLeafSize(LeafSize,LeafSize,LeafSize);
	voxel_grid_source.setInputCloud(source);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter_source (new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_source.filter(*cloud_filter_source);

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *target) == -1)
	{
		PCL_ERROR("Couldn't read file \n");
		return (-1);
	}	
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_target;
	voxel_grid_target.setLeafSize(LeafSize,LeafSize,LeafSize);
	voxel_grid_target.setInputCloud(target);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter_target (new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid_target.filter(*cloud_filter_target);

	std::vector<facenode> face_vecter_source;
	face_extrate(cloud_filter_source,face_vecter_source);
	describ des_source;
	creat_describ(cloud_filter_source,face_vecter_source,des_source);

	std::vector<facenode> face_vecter_target;
	face_extrate(cloud_filter_target,face_vecter_target);
	describ des_target;
	creat_describ(cloud_filter_target,face_vecter_target,des_target);

	float score=compare_describ(des_source,des_target);
	std::cout<<"similarity score: "<<score<<std::endl;
	return 0;			
}