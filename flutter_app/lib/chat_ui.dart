import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'api_service.dart';
import 'cache_handler.dart';

class ChatUI extends StatefulWidget {
  const ChatUI({super.key});
  @override
  State<ChatUI> createState() => _ChatUIState();
}

class _ChatUIState extends State<ChatUI> {
  final _api = ApiService();
  final _cache = CacheHandler();
  final _ctrl = TextEditingController();
  List<Map<String, dynamic>> _messages = [];

  @override
  void initState() {
    super.initState();
    _loadCache();
  }

  void _loadCache() async {
    final cached = await _cache.load();
    setState(() {
      _messages = cached.map((m) => {"q": m["q"], "a": m["a"], "images": []}).toList();
    });
  }

  Future<void> _send() async {
    final q = _ctrl.text.trim();
    if (q.isEmpty) return;
    setState(() => _messages.add({"q": q, "a": "…", "images": []}));
    _ctrl.clear();
    try {
      final res = await _api.ask(q);
      final a = res["answer"] ?? "";
      final imgs = (res["images"] as List?)?.cast<String>() ?? [];
      setState(() {
        _messages[_messages.length - 1] = {"q": q, "a": a, "images": imgs};
      });
      await _cache.save(q, a);
    } catch (e) {
      setState(() => _messages[_messages.length - 1] = {"q": q, "a": "Error: $e", "images": []});
    }
  }

  Uint8List _decodeBase64DataUrl(String dataUrl) {
    final idx = dataUrl.indexOf(",");
    final b64 = idx >= 0 ? dataUrl.substring(idx+1) : dataUrl;
    return base64Decode(b64);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("AI Chat")),
      body: Column(children: [
        Expanded(
          child: ListView.builder(
            itemCount: _messages.length,
            itemBuilder: (_, i) {
              final m = _messages[i];
              final imgs = (m["images"] as List?) ?? [];
              return Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text("Q: ${m["q"]}", style: const TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 6),
                    Text("A: ${m["a"]}"),
                    if (imgs.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Wrap(spacing: 8, children: imgs.map((d) {
                        return Image.memory(_decodeBase64DataUrl(d), width:120, height:120, fit:BoxFit.cover);
                      }).toList())
                    ]
                  ]),
                ),
              );
            },
          ),
        ),
        Row(children: [
          Expanded(child: Padding(padding: const EdgeInsets.all(8), child: TextField(controller: _ctrl, decoration: const InputDecoration(hintText:"Ask in English or Hindi...")))),
          IconButton(icon: const Icon(Icons.send), onPressed: _send),
        ])
      ]),
    );
  }
}
