#!/usr/bin/env python3
"""
模型管理工具
用于管理rf-model文件夹中的训练模型
"""

import os
import json
import pickle
from datetime import datetime
import glob

class ModelManager:
    def __init__(self, model_folder='rf-model', stock_code=None):
        """
        初始化模型管理器
        
        Args:
            model_folder: 模型文件夹路径
            stock_code: 股票代码，如果为None则尝试从配置文件读取
        """
        self.model_folder = model_folder
        
        # 获取股票代码
        if stock_code is None:
            # 尝试从配置文件读取
            try:
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.stock_code = config['data_config']['stock_code']
            except:
                print("⚠️ 无法从配置文件读取股票代码，某些功能可能不可用")
                self.stock_code = None
        else:
            self.stock_code = stock_code
        
        self.ensure_folder_exists()
    
    def ensure_folder_exists(self):
        """确保模型文件夹存在"""
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            print(f"创建模型文件夹: {self.model_folder}")
    
    def list_models(self):
        """列出所有可用的模型"""
        print(f"\n=== {self.model_folder} 文件夹中的模型 ===")
        
        # 查找所有pkl文件
        model_files = glob.glob(f"{self.model_folder}/*.pkl")
        info_files = glob.glob(f"{self.model_folder}/*.json")
        
        if not model_files:
            print("未找到任何模型文件")
            return
        
        print(f"模型文件 ({len(model_files)} 个):")
        for model_file in sorted(model_files):
            filename = os.path.basename(model_file)
            size = os.path.getsize(model_file)
            mtime = datetime.fromtimestamp(os.path.getmtime(model_file))
            print(f"  - {filename} ({size/1024:.1f}KB, {mtime.strftime('%Y-%m-%d %H:%M')})")
        
        print(f"\n信息文件 ({len(info_files)} 个):")
        for info_file in sorted(info_files):
            filename = os.path.basename(info_file)
            print(f"  - {filename}")
    
    def show_model_info(self, info_file=None):
        """显示模型详细信息"""
        # 如果没有指定文件名且有股票代码，使用新格式
        if info_file is None and self.stock_code:
            info_file = f'model_info_{self.stock_code}.json'
        elif info_file is None:
            info_file = 'model_info.json'  # 向后兼容
        
        info_path = os.path.join(self.model_folder, info_file)
        
        if not os.path.exists(info_path):
            print(f"模型信息文件不存在: {info_path}")
            return None
        
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            print(f"\n📊 模型信息 ({info_file}):")
            print("=" * 50)
            
            # 显示基本信息
            basic_fields = ['model_type', 'training_date', 'stock_code', 'stock_name', 'feature_count']
            for field in basic_fields:
                if field in model_info:
                    print(f"{field}: {model_info[field]}")
            
            # 显示模型参数
            if 'model_params' in model_info:
                print(f"\n模型参数:")
                for key, value in model_info['model_params'].items():
                    print(f"  {key}: {value}")
            
            if 'features' in model_info:
                print(f"\n使用特征:")
                for i, feature in enumerate(model_info['features'], 1):
                    print(f"  {i:2d}. {feature}")
            
            return model_info
            
        except Exception as e:
            print(f"读取模型信息失败: {e}")
            return None
    
    def load_model(self, model_file=None):
        """加载模型"""
        # 如果没有指定文件名且有股票代码，使用新格式
        if model_file is None and self.stock_code:
            model_file = f'model_{self.stock_code}.pkl'
        elif model_file is None:
            model_file = 'optimized_model.pkl'  # 向后兼容
        
        model_path = os.path.join(self.model_folder, model_file)
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"模型加载成功: {model_file}")
            print(f"模型类型: {type(model).__name__}")
            
            # 显示模型参数
            if hasattr(model, 'get_params'):
                params = model.get_params()
                print("模型参数:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    def load_features(self, features_file=None):
        """加载特征列表"""
        # 如果没有指定文件名且有股票代码，使用新格式
        if features_file is None and self.stock_code:
            features_file = f'features_{self.stock_code}.pkl'
        elif features_file is None:
            features_file = 'optimized_model_features.pkl'  # 向后兼容
        
        features_path = os.path.join(self.model_folder, features_file)
        
        if not os.path.exists(features_path):
            print(f"特征文件不存在: {features_path}")
            return None
        
        try:
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            
            print(f"特征列表加载成功: {features_file}")
            print(f"特征数量: {len(features)}")
            print("特征列表:")
            for i, feature in enumerate(features, 1):
                print(f"  {i:2d}. {feature}")
            
            return features
            
        except Exception as e:
            print(f"特征列表加载失败: {e}")
            return None
    
    def validate_model(self):
        """验证模型完整性"""
        print(f"\n=== 验证模型完整性 ===")
        
        if self.stock_code:
            # 使用新的文件名格式
            required_files = [
                f'model_{self.stock_code}.pkl',
                f'features_{self.stock_code}.pkl'
            ]
            
            optional_files = [
                f'model_info_{self.stock_code}.json'
            ]
            print(f"检查股票代码: {self.stock_code}")
        else:
            # 向后兼容旧格式
            required_files = [
                'optimized_model.pkl',
                'optimized_model_features.pkl'
            ]
            
            optional_files = [
                'model_info.json'
            ]
            print("使用通用文件名格式")
        
        all_valid = True
        
        for file in required_files:
            file_path = os.path.join(self.model_folder, file)
            if os.path.exists(file_path):
                print(f"✓ {file} - 存在")
            else:
                print(f"✗ {file} - 缺失 (必需)")
                all_valid = False
        
        for file in optional_files:
            file_path = os.path.join(self.model_folder, file)
            if os.path.exists(file_path):
                print(f"✓ {file} - 存在")
            else:
                print(f"⚠ {file} - 缺失 (可选)")
        
        if all_valid:
            print("\n✅ 模型文件完整，可以正常使用")
        else:
            print("\n❌ 模型文件不完整，请重新训练")
        
        return all_valid
    
    def backup_model(self, backup_name=None):
        """备份当前模型"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_folder = os.path.join(self.model_folder, backup_name)
        
        if os.path.exists(backup_folder):
            print(f"备份文件夹已存在: {backup_folder}")
            return False
        
        os.makedirs(backup_folder)
        
        # 复制所有文件到备份文件夹
        import shutil
        files_to_backup = glob.glob(f"{self.model_folder}/*.*")
        
        copied_files = 0
        for file_path in files_to_backup:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                backup_path = os.path.join(backup_folder, filename)
                shutil.copy2(file_path, backup_path)
                copied_files += 1
        
        print(f"✅ 模型已备份到: {backup_folder}")
        print(f"备份文件数量: {copied_files}")
        return True
    
    def clean_folder(self):
        """清理模型文件夹"""
        print(f"\n⚠️ 警告: 即将删除 {self.model_folder} 文件夹中的所有文件")
        confirm = input("确认删除? (输入 'yes' 确认): ")
        
        if confirm.lower() != 'yes':
            print("操作已取消")
            return False
        
        files = glob.glob(f"{self.model_folder}/*.*")
        deleted_count = 0
        
        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"删除: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
        
        print(f"✅ 已删除 {deleted_count} 个文件")
        return True

def main():
    """主函数"""
    print("🔧 模型管理工具")
    print("=" * 50)
    
    manager = ModelManager()
    
    while True:
        print("\n请选择操作:")
        print("1. 列出所有模型")
        print("2. 显示模型详细信息")
        print("3. 加载并查看模型")
        print("4. 加载并查看特征")
        print("5. 验证模型完整性")
        print("6. 备份当前模型")
        print("7. 清理模型文件夹")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-7): ").strip()
        
        if choice == '0':
            print("再见!")
            break
        elif choice == '1':
            manager.list_models()
        elif choice == '2':
            manager.show_model_info()
        elif choice == '3':
            manager.load_model()
        elif choice == '4':
            manager.load_features()
        elif choice == '5':
            manager.validate_model()
        elif choice == '6':
            backup_name = input("输入备份名称 (留空使用默认): ").strip()
            if not backup_name:
                backup_name = None
            manager.backup_model(backup_name)
        elif choice == '7':
            manager.clean_folder()
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 