terraform {
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.220"
    }
  }
}

provider "alicloud" {
  region     = "ap-southeast-1"  # Singapore (same region as DashScope)
  access_key = var.alicloud_access_key
  secret_key = var.alicloud_secret_key
}

variable "alicloud_access_key" {
  type      = string
  sensitive = true
}

variable "alicloud_secret_key" {
  type      = string
  sensitive = true
}

# Security group allowing HTTP and SSH
resource "alicloud_security_group" "agent_sg" {
  name        = "alpha-signal-agent-sg"
  description = "Security group for memory agent backend"
  vpc_id      = alicloud_vpc.agent_vpc.id
}

resource "alicloud_vpc" "agent_vpc" {
  vpc_name   = "alpha-signal-vpc"
  cidr_block = "172.16.0.0/16"
}

resource "alicloud_vswitch" "agent_vswitch" {
  vpc_id     = alicloud_vpc.agent_vpc.id
  cidr_block = "172.16.0.0/24"
  zone_id    = "ap-southeast-1a"
}

resource "alicloud_security_group_rule" "allow_http" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "8000/8000"
  security_group_id = alicloud_security_group.agent_sg.id
  cidr_ip           = "0.0.0.0/0"
}

resource "alicloud_security_group_rule" "allow_ssh" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "22/22"
  security_group_id = alicloud_security_group.agent_sg.id
  cidr_ip           = "0.0.0.0/0"
}

# ECS Instance (free tier eligible)
resource "alicloud_instance" "agent_server" {
  instance_name              = "alpha-signal-memory-agent"
  instance_type              = "ecs.t5-lc1m1.small"  # 1 vCPU, 1GB RAM (free tier)
  image_id                   = "ubuntu_22_04_x64_20G_alibase_20230907.vhd"
  security_groups            = [alicloud_security_group.agent_sg.id]
  vswitch_id                 = alicloud_vswitch.agent_vswitch.id
  internet_max_bandwidth_out = 5  # 5 Mbps public bandwidth
  system_disk_category       = "cloud_efficiency"
  system_disk_size           = 40

  # Startup script to install dependencies
  user_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update && apt-get install -y python3-pip sqlite3 nginx
    pip3 install fastapi uvicorn httpx openai sqlite-utils pydantic
    # Clone and start the agent
    cd /opt
    git clone https://github.com/basilwong/alpha-signal-analysis.git
    cd alpha-signal-analysis
    # Start the backend
    nohup python3 -m uvicorn agent.server:app --host 0.0.0.0 --port 8000 &
  EOF
  )
}

# Allocate public IP
resource "alicloud_eip_address" "agent_eip" {
  bandwidth = 5
}

resource "alicloud_eip_association" "agent_eip_assoc" {
  allocation_id = alicloud_eip_address.agent_eip.id
  instance_id   = alicloud_instance.agent_server.id
  instance_type = "EcsInstance"
}

output "agent_public_ip" {
  value = alicloud_eip_address.agent_eip.ip_address
}

output "agent_instance_id" {
  value = alicloud_instance.agent_server.id
}
