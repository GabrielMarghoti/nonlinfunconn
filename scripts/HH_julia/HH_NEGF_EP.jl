#
# Hodgkin-Huxley Neuron model NEGF
#
# Gabriel Marghoti / July 2024 Shanghai
#
using Plots
using DifferentialEquations
using LaTeXStrings
using ProgressBars
using Statistics

pythonplot()

function conv(X, Y, interval)
    return ((interval[end]-interval[1])/length(interval))*sum(X.*Y)
end

function α_n(V)
    return 0.01 * (V + 55) / (1 - exp(-(V + 55) / 10))
end

function β_n(V)
    return 0.125 * exp(-(V + 65) / 80)
end

# Alpha and Beta functions for the gating variable m
function α_m(V)
    return 0.1 * (V + 40) / (1 - exp(-(V + 40) / 10))
end

function β_m(V)
    return 4 * exp(-(V + 65) / 18)
end

# Alpha and Beta functions for the gating variable h
function α_h(V)
    return 0.07 * exp(-(V + 65) / 20)
end

function β_h(V)
    return 1 / (1 + exp(-(V + 35) / 10))
end

#################################################################################
# Derivatives of alpha and beta functions for the gating variable m
function dα_n(V)
    a, b, c = 0.01, 55, 10
    numerator = a * (1 - exp(-(V + b) / c)) - a * (V + b) * (1 / c) * exp(-(V + b) / c)
    denominator = (1 - exp(-(V + b) / c))^2
    return numerator / denominator
end

function dβ_n(V)
    a, b, c = 0.125, 65, 80
    return -a / c * exp(-(V + b) / c)
end

# Derivatives of alpha and beta functions for the gating variable m
function dα_m(V)
    a, b, c = 0.1, 40, 10
    numerator = a * (1 - exp(-(V + b) / c)) - a * (V + b) * (1 / c) * exp(-(V + b) / c)
    denominator = (1 - exp(-(V + b) / c))^2
    return numerator / denominator
end

function dβ_m(V)
    a, b, c = 4, 65, 18
    return -a / c * exp(-(V + b) / c)
end

# Derivatives of alpha and beta functions for the gating variable h
function dα_h(V)
    a, b, c = 0.07, 65, 20
    return -(a / c) * exp(-(V + b) / c)
end

function dβ_h(V)
    b, c = 35, 10
    return exp(-(V + b) / c) / (c * (1 + exp(-(V + b) / c))^2)
end

##############################################
## Voltage Clamp functions

function n∞(V)
    return α_n(V)/(α_n(V)+β_n(V))
end

function τ_n(V)
    return 1/(α_n(V)+β_n(V))
end

function m∞(V)
    return α_m(V)/(α_m(V)+β_m(V))
end

function τ_m(V)
    return 1/(α_m(V)+β_m(V))
end

function h∞(V)
    return α_h(V)/(α_h(V)+β_h(V))
end

function τ_h(V)
    return 1/(α_h(V)+β_h(V))
end

function dn∞(V)
    return dα_n(V)/(α_n(V)+β_n(V)) - α_n(V)*(dα_n(V)+dβ_n(V))/((α_n(V)+β_n(V))^2)
end

function dm∞(V)
    return dα_m(V)/(α_m(V)+β_m(V)) - α_m(V)*(dα_m(V)+dβ_m(V))/((α_m(V)+β_m(V))^2)
end

function dh∞(V)
    return dα_h(V)/(α_h(V)+β_h(V)) - α_h(V)*(dα_h(V)+dβ_h(V))/((α_h(V)+β_h(V))^2)
end



# Hodgkin-Huxley model equations
function hh_model!(du, u, p, t)
    V, Vm, Vh, Vn = u
    C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, I_ext, stim = p

    
    if t>=1.0 && t<=2.0
        I_ext += stim
    end

    # Ion channel dynamics


    #=
    α_m = 0.182  * (V + 35) / (1 - exp(-(V + 35) / 9))
    β_m = -0.124 * (V + 35) / (1 - exp((V + 35) / 9))
    
    α_h = 0.25 * exp(-(V + 90) / 12)
    β_h = 0.25 * exp((V + 62) / 6) / exp((V + 90) / 12)

    α_n = 0.02   * (V - 25) / (1 - exp(-(V - 25) / 9))
    β_n = -0.002 * (V - 25) / (1 - exp((V - 25) / 9))
    =#

    du[1] = (I_ext - g_Na_bar*m∞(Vm)^3*h∞(Vh)*(V - E_Na) - g_K_bar*n∞(Vh)^4*(V - E_K) - g_L_bar*(V - E_L)) / C_m
    du[2] = (m∞(V)-m∞(Vm))/(τ_m(V)*dm∞(Vm))
    du[3] = (h∞(V)-h∞(Vh))/(τ_h(V)*dh∞(Vh))
    du[4] = (n∞(V)-n∞(Vn))/(τ_n(V)*dn∞(Vn))
end


function main()
        
    # Initial conditions: V, m, h, n
    V0, m0, h0, n0 = -65.0, -10*rand(), -10*rand(), -10*rand()
    u0 = [V0, m0, h0, n0]

    # Time span
    ti =   0.0
    tf =  100.0
    
    resolution = 2000 

    tspan = (ti, tf)

    tₛ = range(ti, tf, length=resolution)

    # Parameters

    # Define constants
    C_m      =   1.0         # membrane capacitance, in uF/cm^2
    g_Na_bar = 120.0         # maximum conductances, in mS/cm^2
    g_K_bar  =  36.0
    g_L_bar  =   0.3
    E_Na     =  50.0         # Nernst reversal potentials, in mV
    E_K      = -77.0
    E_L      = -55.0 

    # External current
    I_ext = 0.0  # in μA/cm^2
    stim  = 10.0  # in μA/cm^2

    figures_path = "/home/gabriel/figuras/qualification/NEGF_HH_EP/$(C_m)_$(g_Na_bar)_$(g_K_bar)_$(g_L_bar)_$(E_Na)_$(E_K)_$(E_L)_$(I_ext)_$(stim)/"
    mkpath(figures_path)

    # Solve the differential equations TRANSIENT
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, I_ext, 0.0])
    sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    png(plot(sol, layout=(4,1), lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", frame_style=:box, size=(500,500), dpi=200), figures_path*"HH_variables_simulation_transient")


    V_eq  = sol[end][1] 
    Vm_eq = sol[end][2]
    Vh_eq = sol[end][3]
    Vn_eq = sol[end][4]

    u0 = sol[end]
    V0, m0, h0, n0 = u0

    # Solve the differential equations STIMULATED
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, I_ext, stim])
    sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    png(plot(sol, layout=(4,1), 
    lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "Vₘ(t)" "Vₕ(t)" "Vₙ(t)"], label="", 
    frame_style=:box, size=(500,500), dpi=200, grid=false),
     figures_path*"HH_variables_simulation")

    V = zeros(resolution)
    Vm = zeros(resolution)
    Vh = zeros(resolution)
    Vn = zeros(resolution)
    
    for t=1:resolution
        V[t] = sol[t][1] 
        Vm[t] = sol[t][2]
        Vh[t] = sol[t][3]
        Vn[t] = sol[t][4]
    end

    n_inf = α_n.(V) ./ (α_n.(V) .+ β_n.(V))
    tau_n = 1. ./ (α_n.(V) .+ β_n.(V))

    n_inf_Vn  =   n∞.(Vn)
    dn_inf_Vn =  dn∞.(Vn)

    m_inf = α_n.(V) ./ (α_n.(V) .+ β_n.(V))
    tau_m = 1. ./ (α_n.(V) .+ β_n.(V))

    h_inf = α_n.(V) ./ (α_n.(V) .+ β_n.(V))
    tau_h = 1. ./ (α_n.(V) .+ β_n.(V))

    vi_range = range(-95,45, length=resolution)

    n_func_vi =  α_n.(vi_range) ./ (α_n.(vi_range) .+ β_n.(vi_range))
    m_func_vi =  α_m.(vi_range) ./ (α_m.(vi_range) .+ β_m.(vi_range))
    h_func_vi =  α_h.(vi_range) ./ (α_h.(vi_range) .+ β_h.(vi_range))


    png(
        plot([vi_range vi_range vi_range], [n_func_vi m_func_vi h_func_vi],
            xflip=false,
            xlabel = L"𝒗_𝒙",
            label= [L"𝒏" L"𝒎" L"𝒉"],
            ylabel = L"𝒙",
            title="Equivalent Potentials",
            size=(500,400),
            lc=[:black :red :blue],

            ls=:solid,
            dpi=200, 
            frame_style=:box,
            legend=:right,
            grid=false),
            figures_path*"equivalent_potential"
        )

    tau_n_const = mean(tau_n)

    png(plot(tₛ, [V (g_Na_bar*m∞.(Vm).^(3).*h∞.(Vh)) (g_K_bar*n∞.(Vn).^4)], layout=(3,1), 
    lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"V(t)\ (mV)" L"g_{Na} (t)" L"g_{K} (t)"], label="", 
    frame_style=:box, size=(500,500), dpi=200, grid=false),
     figures_path*"HH_conductances_simulation")


#   NONequilibrium direct green functions#
    σ_n_V  = zeros(resolution, resolution)
    σ_n_V_const_tau  = zeros(resolution, resolution)
    Γ_gK_V = zeros(resolution, resolution)

    conv_sigma_n_V = zeros(resolution) # n
    conv_sigma_n_V_const_tau = zeros(resolution) # n approx
    for itr=ProgressBar(1:60)  #### iterative method to approximate σ

        for t=1:resolution
            for t′=1:(t-1) 
                σ_n_V[t, t′]  = (n_inf[t′] - (n0+n_inf_Vn[t′])) / ((V[t′].-V_eq)*tau_n[t′]*dn_inf_Vn[t′])
                σ_n_V_const_tau[t, t′]  = (exp(-(t-t′)/tau_n_const)*n_inf[t′]) / ((V[t′].-V_eq)*tau_n_const)
            end
        end
                  
        for t=1:resolution
            conv_sigma_n_V[t] = conv(σ_n_V[t, 1:t], (V[1:t].-V_eq), tₛ[1:t])# ones(length(1:t)), tₛ[1:t])
            conv_sigma_n_V_const_tau[t] = conv(σ_n_V_const_tau[t, 1:t], (V[1:t].-V_eq), tₛ[1:t])# ones(length(1:t)), tₛ[1:t])
        end
                    
        if itr%10==0
            png(
                heatmap(tₛ, tₛ, σ_n_V,
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        size=(500,400),
                        title=L"σ_{n,V}(t,t′)",
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        clims = (-1000,1000),
                        grid=false),
                        figures_path*"sigma_n_V"
                )

            png(
                plot(tₛ[end] .- tₛ, [σ_n_V[end, 1:end] σ_n_V_const_tau[end, 1:end]],
                        xflip=false,
                        ylabel = "NEGF",
                        label= [L"σ_{n,V}(t,t′)" L"̃σ_{n,V}(t,t′)"],
                        xlabel = "t - t′",
                        title="t=$(tₛ[end])",
                        size=(1000,800),
                        lc=[:black :red],
                        ls=[:solid :dash],
                        dpi=200, 
                        frame_style=:box,
                        ylims = (-1000,1000),
                        grid=false),
                        figures_path*"level_curve_sigma_n_V"
                )
                      png(
                plot(V[end] .- V, [σ_n_V[end, 1:end] σ_n_V_const_tau[end, 1:end]],
                        xflip=false,
                        ylabel = "NEGF",
                        label= [L"σ_{n,V}(t,t′)" L"̃σ_{n,V}(t,t′)"],
                        xlabel = "t - t′",
                        title="t=$(tₛ[end])",
                        size=(1000,800),
                        lc=[:black :red],
                        ls=[:solid :dash],
                        dpi=200, 
                        frame_style=:box,
                        ylims = (-400,400),
                        grid=false),
                        figures_path*"curve_VxV_sigma_n_V"
                )

            plot_conv_n_V = plot(tₛ, [Vn n0.+conv_sigma_n_V],     
            label=["Original" "Estimated"],
            size=(500,400),
            lc = [:black :red],
            ls = [:solid :dash],
            xlabel="Time (ms)",
            ylabel = "n",
            dpi=200, 
            frame_style=:box,
            #ylims=(-0.01, 1.01),
            grid=false)
            png(plot_conv_n_V,       
                figures_path*"plot_n_simulated_estimated_itr$(itr)"
            )
        end
    end
#


#=
####################################################################################
#### Equilibrium direct green functions
    σ₀  = zeros(resolution, resolution, N, N)
    gg₀ = zeros(resolution, resolution, N, N)
    gs₀ = zeros(resolution, resolution, N, N)
    g₀  = zeros(resolution, resolution, N, N)

    for j=1:N
        for i=j:N
            for t′=1:resolution
                σ₀[:, t′, i, j]  = Θ.(tₛ.-tₛ[t′]) .*a_r[i,j]*(1-Seq[i, j]).*dΦ(Veq[j], β[i,j], V_th[i,j]).*exp.(-(tₛ.-tₛ[t′]).*(a_d[i,j]-a_r[i,j]/(1 + exp(-β[i,j]*(Veq[j]-V_th[i,j])))))
                gg₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*γg[i, j].*exp.(-(tₛ.-tₛ[t′]).*(γ[i].+sum(γg[i, :]).+sum(γs[i, :].*Seq[i, :])))
                gs₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*γs[i, j].*(Es[i,j]-Veq[i]).*exp.(-(tₛ.-tₛ[t′]).*(γ[i].+sum(γg[i, :]).+sum(γs[i, :].*Seq[i, :])))
            end
            png(
                heatmap(tₛ, tₛ, σ₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                figures_path*"sigmaZERO_ij$(i)_$(j)"
            )

            png(heatmap(tₛ, tₛ, gs₀[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"gsZERO_ij$(i)_$(j)"
            )

            png(heatmap(tₛ, tₛ, gg₀[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"ggZERO_ij$(i)_$(j)"
            )
        end
    end

    for t=1:resolution
        for t′=1:(t-1)
            for i=1:N
                for j=1:N
                    g₀[t, t′, i, j] = gg₀[t, t′, i, j] + conv(gs₀[t, t′:t, i, j], σ₀[t′:t, t′, i, j], tₛ[t′:t])
                end
            end
        end
    end
    
##############################################################################################
#### Integrate the system to get the ΔV and ΔS deviations from the equilibrium

    p = [N, γ, Ec, γg, γs, Es, a_r, a_d, β, V_th, I, 0.0, -1.0, C]

    ds = ODEProblem(c_elegans_model, u0, (0.0, tf), p)
    tr = solve(ds, Tsit5(), dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    Vs = zeros(resolution, N)
    Ss = zeros((resolution, N, N))

    ΔVs = zeros(resolution, N)
    ΔSs = zeros((resolution, N, N))

    for t=1:resolution
        Vs[t, :] = tr[t][:, 1]
        ΔVs[t, :] = tr[t][:, 1] .- Veq
        Ss[t, :, :] = tr[t][:, 2:end]
        ΔSs[t, :, :] = tr[t][:, 2:end] .- Seq
    end

    # Plot variables trajectory
    png(
        plot(tₛ, ΔVs,
            label=["N1" "N2" "N3" "N4"],
            xlabel="time (ms)",
            ylabel = "ΔV",
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false),
            figures_path*"potentials_duration_stim"
        )
#

###########################################################################
#### NONequilibrium direct green functions#
    σ = zeros(resolution, resolution, N, N)
    𝜋 = zeros(resolution, resolution, N, N)
    g = zeros(resolution, resolution, N, N)


    conv_sigma_V = zeros(resolution, N, N) # ΔS
    for itr=1:3  #### iterative method to approximate σ
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)   
                        if Vs[t′, j] == Veq[j]
                            σ[t, t′, i, j]  = 0.0 #(σ₀[t, t′, i, j]/dΦ(Veq[j], β[i, j], V_th[i, j]))*0*(1 - (Ss[t′, i, j]-Seq[i, j])/(1-Seq[i, j]))
                        else
                            σ[t, t′, i, j]  = (σ₀[t, t′, i, j]/dΦ(Veq[j], β[i, j], V_th[i, j]))*((Φ(Vs[t′, j], β[i, j], V_th[i,j])-Φ(Veq[j], β[i, j], V_th[i,j]))/ΔVs[t′, j])*(1 - (conv_sigma_V[t′, i, j])/(1-Seq[i, j]))
                        end
                    end
                end
                for t=1:resolution
                    conv_sigma_V[t, i, j] = conv(σ[t, 1:t, i, j], ΔVs[1:t, j], tₛ[1:t])
                end
                        
            for t=1:resolution
                for t′=1:(t-1) 
                    𝜋[t, t′, i, j] = conv(gs₀[t, t′:t, i, j], (1 .-(ΔVs[t′:t, i]/(Es[i, j]-Veq[i]))).*σ[t′:t, t′, i, j], tₛ[t′:t])
                    g[t, t′, i, j] = gg₀[t, t′, i, j] + 𝜋[t, t′, i, j]
                end
            end
                    
                png(
                    heatmap(tₛ, tₛ, σ[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="σ_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"sigma_ij$(i)_$(j)"
                    )

                png(
                    heatmap(tₛ, tₛ, g[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="g_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"g_ij$(i)_$(j)"
                    )
                png(
                    heatmap(tₛ, tₛ, 𝜋[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="𝜋_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"𝜋_ij$(i)_$(j)"
                    )
            end
        end
        plot_Delta_S_sigma_V = plot(tₛ, [ΔSs[:, 2, 1] ΔSs[:, 3, 2] ΔSs[:, 4, 1]  ΔSs[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label=["S [2←1]" "S [3⇐2]" "S [4⥒1]" "S [4←3]"],
        size=(500,400),
        lc = [:gray :black :red :blue],
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_sigma_V[:, 2, 1] conv_sigma_V[:, 3, 2] conv_sigma_V[:, 4, 1]  conv_sigma_V[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="",
        lc = [:gray :black :red :blue],
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false),
        png(plot_Delta_S_sigma_V,       
            figures_path*"synapses_duration_stim_itr$(itr)"
        )
    end


        plot_Delta_S_sigma_V = plot(tₛ, [ΔSs[:, 2, 1] ΔSs[:, 3, 2] ΔSs[:, 4, 1]  ΔSs[:, 4, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label=["S [2←1]" "S [3⇐2]" "S [4⥒1]" "S [4←3]"],
        lc = [:gray :black :red :blue],
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_sigma_V[:, 2, 1] conv_sigma_V[:, 3, 2] conv_sigma_V[:, 4, 1]  conv_sigma_V[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="",
        lc = [:gray :black :red :blue],
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false),
    png(plot_Delta_S_sigma_V,       
        figures_path*"synapses_duration_stim"
    )
   
#######################################################################################
##### CONNECTED Green functions
#####    
    # The connected Green function is the directed plus the convolution with the paths from node j to node i
    # First compute the trivial ones, the first neighbors of the boundary condition ΔVⱼ, then propagate to higher neighbors 
    G₀ = g₀                # First neighbors (direct)       
    
    conv_G_V = zeros(resolution, N) # test G function
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            G₀[t, t′, 3, 1] += conv(g₀[t, t′:t, 3, 2], G₀[t′:t, t′, 2, 1], tₛ[t′:t])
            G₀[t, t′, 4, 2] += conv(g₀[t, t′:t, 4, 3], G₀[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            G₀[t, t′, 4, 1] += conv(g₀[t, t′:t, 4, 3], G₀[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end

    G = g                # First neighbors (direct)                        
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            G[t, t′, 3, 1] += conv(g[t, t′:t, 3, 2], G[t′:t, t′, 2, 1], tₛ[t′:t])
            G[t, t′, 4, 2] += conv(g[t, t′:t, 4, 3], G[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            G[t, t′, 4, 1] += conv(g[t, t′:t, 4, 3], G[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end
    for i=1:N
        for t=1:resolution
            for j=1:N
                conv_G_V[t, i] += conv(g[t, 1:t, i, j], ΔVs[1:t, j], tₛ[1:t])
            end
        end
    end


    for j=1:N-1
        for i=j:N
            png(
                heatmap(tₛ, tₛ, G₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    title="G₀$(i),$(j)(t,t′)",
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"G0_ij$(i)_$(j)"
                )
            png(
                heatmap(tₛ, tₛ, G[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    title="G$(i),$(j)(t,t′)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"G_ij$(i)_$(j)"
                )

            plote_level_curves_G = plot(tₛ[end].-tₛ[1:end], G₀[end, 1:end, i, j],
                            xlabel="t-t′",
                            ylabel = "G",
                            label= "G₀",
                            lc=:black,
                            lw=2.4,
                            size=(500,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
            for t_idx in probe_times_idx
                plot!(tₛ[t_idx].-tₛ[2:(t_idx)], G[t_idx, 2:(t_idx), i, j],
                            xlabel="t-t′",
                            ylabel = "G",
                            lc = RGBA(t_idx/plot_times_idx[end], 0, 1 - (2*t_idx/plot_times_idx[end] -1)^2, 1),
                            la=0.9,
                            lw=1.4,
                            ls= :solid, #rand([:dash; :dot]),
                            label= "t = "*string(round(tₛ[t_idx], digits=2)),
                            size=(500,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
                    png(plote_level_curves_G,
                            figures_path*"G_level_curves_ij$(i)_$(j)"
                        )
            end
        end
    end

      # Plot variables trajectory
      plot_DV_GDV = plot(tₛ, ΔVs,
            label=["N1" "N2" "N3" "N4"],
            xlabel="time (ms)",
            lc = [:gray :black :red :blue],
            ylabel = "ΔV",
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false)
        plot!(tₛ, conv_G_V,
            label="",
            xlabel="time (ms)",
            ls=:dot,
            lc = [:gray :black :red :blue],
            ylabel = "ΔV",
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false)
      png(
        plot_DV_GDV,
            figures_path*"potentials_duration_stim"
        )

#################################################################################################
####### Response Synaptic Susceptibility

    χ     = zeros(resolution, resolution, N, N)  

    conv_sigma0_V_chi = zeros(resolution, resolution, N, N)

    for itr=1:3
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)    
                        χ[t, t′, i, j]  = ((σ₀[t, t′, i, j]*dΦ(Vs[t′, j], β[i, j], V_th[i, j]))/dΦ(Veq[j], β[i, j], V_th[i, j]))*(1 - (conv_sigma_V[t′, i, j])/(1-Seq[i, j])) - conv_sigma0_V_chi[t, t′, i, j]*(1/(dΦ(Veq[j], β[i, j], V_th[i, j])*(1-Seq[i,j])))   
                    end
                end
                for t=1:resolution
                    for t′=1:(t-1)    
                        conv_sigma0_V_chi[t, t′, i, j] = conv(σ₀[t, t′:t, i, j].*(Φ.(Vs[t′:t, j], β[i, j], V_th[i,j]).-Φ(Veq[j], β[i, j], V_th[i,j])), χ[t′:t, t′, i, j], tₛ[t′:t])
                    end
                end

            png(
                heatmap(tₛ, tₛ, χ[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="χ_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"Chi_ij$(i)_$(j)_itr$(itr)"
                    )
            end
        end
    end

################################################################
#### Direct response function

    f     = zeros(resolution, resolution, N, N)

    for itr=1:3
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)
                        f[t, t′, i, j] = gg₀[t, t′, i, j]  + conv(gs₀[t, t′:t, i, j], (1 .-(ΔVs[t′:t, i]./(Es[i, j]-Veq[i]))).*χ[t′:t, t′, i, j] .- conv_sigma_V[t′:t, i, j].*f[t′:t, t′, i, j], tₛ[t′:t])
                    end
                end

                png(
                    heatmap(tₛ, tₛ, f[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="f_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"f_ij$(i)_$(j)_itr$(itr)"
                    )
            end
        end
    end
                
##################################################################################################
#### Connected response functions

    F = f                # First neighbors (direct)         
                   
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            F[t, t′, 3, 1] += conv(f[t, t′:t, 3, 2], f[t′:t, t′, 2, 1], tₛ[t′:t])
            F[t, t′, 4, 2] += conv(f[t, t′:t, 4, 3], f[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            F[t, t′, 4, 1] += conv(f[t, t′:t, 4, 3], F[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end

    for j=1:N
        for i=j:N

        png(
            heatmap(tₛ, tₛ, F[:, :, i, j],
                xflip=false,
                ylabel="t (current time)",
                xlabel = "t′ (past time)",
                title="F$(i),$(j)(t,t′)",
                size=(500,400),
                fillcolor=:jet1,
                dpi=200, 
                frame_style=:box,
                grid=false),
                figures_path*"F_ij$(i)_$(j)"
            )
        end
    end
    
    plote_level_curves_F = plot(tₛ[end].-tₛ[1:end], G₀[end, 1:end, 4, 1],
            xlabel="t-t′",
            ylabel = "F",
            label= "F₀",
            lc=:black,
            lw=2.4,
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false)
    for t_idx in plot_times_idx
        plot!(tₛ[t_idx].-tₛ[2:(t_idx)], F[t_idx, 2:(t_idx), 4, 1],
                        xlabel="t-t′",
                        ylabel = "F",
                        la=0.9,
                        lw=1.4,
                        lc = RGBA(t_idx/plot_times_idx[end], 0, 1 - (2*t_idx/plot_times_idx[end] -1)^2, 1),
                        ls=:solid, #rand([:dash; :dot]),
                        label= "t = "*string(round(tₛ[t_idx], digits=2)),
                        size=(500,400),
                        dpi=200, 
                        frame_style=:box,
                        grid=false)
    end
    png(plote_level_curves_F,
        figures_path*"F_level_curves_4_1"
        )
    


    # Simulate the system stimulated by strong current I and small probes i
#=
    𝛿Vs = zeros(length(probe_times_idx), resolution, N)
    𝛿Ss = zeros(length(probe_times_idx), resolution, N, N)

    for idx=1:length(probe_times_idx)
        ds = ODEProblem(c_elegans_model, u0, (0.0, tf), [N, γ, Ec, γg, γs, Es, a_r, a_d, β, V_th, I, i, tₛ[probe_times_idx[idx]], C])
        tr = solve(ds, Tsit5(), dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)
    
        for t=1:resolution
            𝛿Vs[idx, t, :] = tr[t][:, 1] .- Vs[t, :]
            𝛿Ss[idx, t, :, :] = tr[t][:, 2:end] .- Ss[t, :, :]
        end
    end
=#
=#

end
main()
