__global__ void MoverKernel(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < part->nop){
        // auxiliary variables
        FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
        FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
        FPpart omdtsq, denom, ut, vt, wt, udotb;
        
        // local (to the particle) electric and magnetic field
        FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
        
        // interpolation densities
        int ix,iy,iz;
        FPfield weight[2][2][2];
        FPfield xi[2], eta[2], zeta[2];
        
        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;



        xptilde = part->x[i];
        yptilde = part->y[i];
        zptilde = part->z[i];
        // calculate the average velocity iteratively
        for(int innter=0; innter < part->NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
            iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
            iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
            
            // calculate weights
            xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
            eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
            zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
            xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
            eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
            zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                        Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                        Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                        Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                        Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                        Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                    }
            
            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= part->u[i] + qomdt2*Exl;
            vt= part->v[i] + qomdt2*Eyl;
            wt= part->w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            part->x[i] = xptilde + uptilde*dto2;
            part->y[i] = yptilde + vptilde*dto2;
            part->z[i] = zptilde + wptilde*dto2;
            
            
        } // end of iteration
        // update the final position and velocity
        part->u[i]= 2.0*uptilde - part->u[i];
        part->v[i]= 2.0*vptilde - part->v[i];
        part->w[i]= 2.0*wptilde - part->w[i];
        part->x[i] = xptilde + uptilde*dt_sub_cycling;
        part->y[i] = yptilde + vptilde*dt_sub_cycling;
        part->z[i] = zptilde + wptilde*dt_sub_cycling;
        
        
        //////////
        //////////
        ////////// BC
                                    
        // X-DIRECTION: BC particles
        if (part->x[i] > grd->Lx){
            if (param->PERIODICX==true){ // PERIODIC
                part->x[i] = part->x[i] - grd->Lx;
            } else { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = 2*grd->Lx - part->x[i];
            }
        }
                                                                    
        if (part->x[i] < 0){
            if (param->PERIODICX==true){ // PERIODIC
                part->x[i] = part->x[i] + grd->Lx;
            } else { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = -part->x[i];
            }
        }
            
        
        // Y-DIRECTION: BC particles
        if (part->y[i] > grd->Ly){
            if (param->PERIODICY==true){ // PERIODIC
                part->y[i] = part->y[i] - grd->Ly;
            } else { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = 2*grd->Ly - part->y[i];
            }
        }
                                                                    
        if (part->y[i] < 0){
            if (param->PERIODICY==true){ // PERIODIC
                part->y[i] = part->y[i] + grd->Ly;
            } else { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = -part->y[i];
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (part->z[i] > grd->Lz){
            if (param->PERIODICZ==true){ // PERIODIC
                part->z[i] = part->z[i] - grd->Lz;
            } else { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = 2*grd->Lz - part->z[i];
            }
        }
                                                                    
        if (part->z[i] < 0){
            if (param->PERIODICZ==true){ // PERIODIC
                part->z[i] = part->z[i] + grd->Lz;
            } else { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = -part->z[i];
            }
        }



    }

}





/** particle mover */
int mover_PC(struct particles* h_part, struct EMfield* h_field, struct grid* h_grd, struct parameters* h_param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< h_param->n_sub_cycles << " - species " << h_part->species_ID << " ***" << std::endl;
    
    particles *part_cpu = &h_part;
    mover_pc_cpu(part_cpu, h_field, h_grd, h_param);

    // start subcycling
    for (int i_sub=0; i_sub <  h_part->n_sub_cycles; i_sub++){



        int ThreadsPerBlock = 256;
        int BlocksPerGrid = (h_part->nop + ThreadsPerBlock - 1)/ThreadsPerBlock;

        particles  *part;
        EMfield   *field;
        grid    *grd;
        parameters *param;


        cudaMalloc( &part  , sizeof(particles));
        cudaMalloc( &field , sizeof(EMfield));
        cudaMalloc( &grd   , sizeof(grid));
        cudaMalloc( &param , sizeof(parameters));

        cudaMemcpy( part,  h_part,  sizeof(particles),  cudaMemcpyHostToDevice);
        cudaMemcpy( field, h_field, sizeof(EMfield),    cudaMemcpyHostToDevice);
        cudaMemcpy( grd,   h_grd,   sizeof(grid),       cudaMemcpyHostToDevice);
        cudaMemcpy( param, h_param, sizeof(parameters), cudaMemcpyHostToDevice);
    
        
        // move each particle with new fields
        MoverKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(part, field, grd, param);

        cudaMemcpy( h_part,  part,  sizeof(particles),  cudaMemcpyDeviceToHost);
        cudaMemcpy( h_field, field, sizeof(EMfield),    cudaMemcpyDeviceToHost);
        cudaMemcpy( h_grd,   grd,   sizeof(grid),       cudaMemcpyDeviceToHost);
        cudaMemcpy( h_param, param, sizeof(parameters), cudaMemcpyDeviceToHost);

        cudaFree(part);
        cudaFree(field);
        cudaFree(grd);
        cudaFree(param);
        

    } // end of one particle

    bool result = comparePart(part_cpu, h_part);
    std::cout << "***  comparing result "<< result << std::endl;

                                                                        
    return(0); // exit succcesfully
} // end of the mover
