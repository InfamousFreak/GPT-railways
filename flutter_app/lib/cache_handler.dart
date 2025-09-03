import 'package:shared_preferences/shared_preferences.dart';

class CacheHandler {
  static const String _key = "chat_history_v2";

  Future<void> save(String q, String a) async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList(_key) ?? [];
    list.add("$q|||$a");
    final trimmed = list.length > 2 ? list.sublist(list.length - 2) : list;
    await prefs.setStringList(_key, trimmed);
  }

  Future<List<Map<String,String>>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList(_key) ?? [];
    return list.map((s) {
      final parts = s.split("|||");
      return {"q": parts[0], "a": parts.length > 1 ? parts[1] : ""};
    }).toList();
  }
}
